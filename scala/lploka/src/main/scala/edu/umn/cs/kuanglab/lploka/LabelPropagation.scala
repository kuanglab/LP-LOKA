package edu.umn.cs.kuanglab.lploka

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import scala.util.Random
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.Vectors

class LabelPropagation {

  var blockSize = 128
  var partitions = 24
  
  def setBlockSize(blockSize: Int) {
    this.blockSize = blockSize
  }
  
  def setPartitions(partitions: Int) {
    this.partitions = partitions
  }
  
  def execute(sc: SparkContext, s: BlockMatrix, f0: CoordinateMatrix, n: Int, alpha: Double, thr: Double, maxIter: Int) : BlockMatrix = {

    var shouldBreak = false

    var f: BlockMatrix = new IndexedRowMatrix(f0.entries.map(e => new IndexedRow(e.i, Vectors.dense((1.0/n))))).toBlockMatrix(blockSize, blockSize)
    
    var f02 = new CoordinateMatrix(f0.entries.repartition(partitions).cache().map(e => 
      new MatrixEntry(e.i, e.j, (1-alpha)*e.value)), f0.numRows, 1).toBlockMatrix(blockSize,blockSize)
    
    for (iter <- 1 to maxIter) {
      printf("LRLP Iteration %d\n", iter)
    
      if (!shouldBreak) {
        
        var fOld = f

        var tmp = s.transpose.multiply(f)
        tmp = s.multiply(tmp)
        tmp = new CoordinateMatrix(tmp.toCoordinateMatrix.entries.repartition(partitions).cache().map(e => 
          new MatrixEntry(e.i, e.j, alpha*e.value)), tmp.numRows, 1).toBlockMatrix(blockSize,blockSize)
        f = tmp.add(f02)
        
        var v1 = f.transpose.toIndexedRowMatrix().rows.repartition(partitions).first().vector.toArray
        var v2 = fOld.transpose.toIndexedRowMatrix().rows.repartition(partitions).first().vector.toArray
        
        var diff = v1.zip(v2).map(e => 
          scala.math.abs(e._1 - e._2)).max
        
        printf("LRLP Iteration %d max diff:%f\n", iter, diff)
          
        if (diff < thr) {
          shouldBreak = true
        }      
        
      }
    }

    return f
    
  }
  
  
}