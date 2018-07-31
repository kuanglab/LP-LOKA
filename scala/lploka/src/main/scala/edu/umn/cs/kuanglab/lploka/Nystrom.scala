package edu.umn.cs.kuanglab.lploka

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix, BlockMatrix, IndexedRowMatrix, IndexedRow, RowMatrix}
import org.apache.spark.mllib.linalg.{Matrix, Matrices}
import breeze.linalg.DenseVector

class Nystrom {

  var blockSize = 128
  var partitions = 24
  
  def setBlockSize(blockSize: Int) {
    this.blockSize = blockSize
  }
  
  def setPartitions(partitions: Int) {
    this.partitions = partitions
  }
  
  def execute(sc: SparkContext, S: CoordinateMatrix, k: Int) : CoordinateMatrix = {

    //var fCoordinateEntries = f.entries.repartition(partitions).cache()
    val Srows = S.toIndexedRowMatrix().rows.repartition(partitions).cache();
    
    // make W block symmetric
    var Wrows = Srows.filter(e => e.index < k)
    val WrowsWithIndex = Wrows.map(e => (e.index, e))
    val WtRowsWithIndex = new IndexedRowMatrix(Wrows)
      .toCoordinateMatrix()
      .transpose()
      .toIndexedRowMatrix().rows.map(e => (e.index, e))
    Wrows = WrowsWithIndex.join(WtRowsWithIndex).map(e => {
      new IndexedRow(
          e._1, 
          Vectors.dense((e._2._1.vector.toArray, e._2._2.vector.toArray).zipped.map(_ + _).map(e2 => e2/2))) 
    })
    
    // eigenvalue adjustment
    val C12rows = Srows.filter(e => e.index >= k);
    val Crows = Wrows.union(C12rows)
    var s = new IndexedRowMatrix(Crows).toCoordinateMatrix().entries.map(e => (e.j, e.value)).reduceByKey((x,y) => Math.abs(x) + Math.abs(y))
    val Wentries = new IndexedRowMatrix(Wrows).toCoordinateMatrix().entries.map(e => (e.i, e)).join(s).map(e => {
      if (e._2._1.i == e._2._1.j) {
        new MatrixEntry(e._2._1.i, e._2._1.j, e._2._2 - e._2._1.value)
      } else {
        e._2._1
      }
    })
    Wrows = new CoordinateMatrix(Wentries).toIndexedRowMatrix().rows;
    
    // SVD of W
    val svd = new IndexedRowMatrix(Wrows).computeSVD(k, computeU = true)
    var a = svd.s.toArray
    var u = svd.U

    // find Sigma_k pseudo-inverse, and take square root
    var invA = a.map(e => scala.math.pow(e,-0.5))
    var u2 = u.toBlockMatrix(blockSize,blockSize)

    var F = new IndexedRowMatrix(Wrows.union(C12rows))
      .toBlockMatrix(blockSize, blockSize)
      .multiply(u2)
      .multiply(new CoordinateMatrix(sc.parallelize(invA.zipWithIndex.map({case (e,i) => 
        new MatrixEntry(i,i,e)
        })), invA.length, invA.length).toBlockMatrix(blockSize,blockSize)).toCoordinateMatrix()

    val s2 = new CoordinateMatrix(F.entries.map(e => (e.j, e.value))
        .reduceByKey(_ + _).map(e => new MatrixEntry(e._1, 1, e._2)))
    val s3 = F.toBlockMatrix(blockSize, blockSize).multiply(s2.toBlockMatrix(blockSize, blockSize)).toCoordinateMatrix().entries.map(e => {
      if (e.value == 0) {
        new MatrixEntry(e.i, e.j, 1)
      } else {
        new MatrixEntry(e.i, e.j, Math.abs(e.value))
      }
    }).map(e => (e.i, e.value))
    

    F = new CoordinateMatrix(F.entries.map(e => (e.i, e)).join(s3).map(e => {
      new MatrixEntry(e._2._1.i, e._2._1.j, e._2._1.value / Math.sqrt(e._2._2))
    }))
    
//    println(s3.toLocalIterator.)
//    F.toIndexedRowMatrix().rows.toLocalIterator.foreach(e => {
//      println(e)
//    })
        
    return F

  }
  
}