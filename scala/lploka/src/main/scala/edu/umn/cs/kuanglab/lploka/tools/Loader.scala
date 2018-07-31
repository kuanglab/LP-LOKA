package edu.umn.cs.kuanglab.lploka.tools

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix, BlockMatrix, IndexedRowMatrix, IndexedRow}

object Loader {
  
  def readCoordinateMatrixFromTxt(sc: SparkContext, filename: String, delimiter: String) : CoordinateMatrix = {
    var obs = sc.textFile(filename)
    var v = obs.flatMap(f => 
      f.split("\n")).zipWithIndex.flatMap({case (line,i) => 
        line.split(delimiter).map(el => el.toDouble).zipWithIndex.map({case (el,j) => 
          new MatrixEntry(i,j,el)
        })
      })
    return new CoordinateMatrix(v)
  }
  
  def readIndexedRowMatrixFromTxt(sc: SparkContext, filename: String, delimiter: String) : IndexedRowMatrix = {
    var obs = sc.textFile(filename)
    var v = obs.flatMap(f => 
      f.split("\n")).zipWithIndex.map({case (line,i) => 
        new IndexedRow(i, Vectors.dense(line.split(delimiter).map(el  => el.toDouble)))
      })
    return new IndexedRowMatrix(v)
  }
  
  def readBlockMatrixFromTxt(sc: SparkContext, filename: String, delimiter: String, blockSize: Int) : BlockMatrix = {
    return readCoordinateMatrixFromTxt(sc, filename, delimiter).toBlockMatrix(blockSize,blockSize)
  }
  
}