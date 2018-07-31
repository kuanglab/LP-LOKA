package edu.umn.cs.kuanglab

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import edu.umn.cs.kuanglab.lploka.tools.Loader
import edu.umn.cs.kuanglab.lploka.Nystrom
import edu.umn.cs.kuanglab.lploka.LabelPropagation

object LPLOKAMain {
  
  def main(args: Array[String]) = {
    
    // creating spark context    
    val conf = new SparkConf().setAppName("LPLOKASpark").setMaster("local")
    val sc = new SparkContext(conf)
    
    // testing nystrom
    val f = Loader.readCoordinateMatrixFromTxt(sc, "data/X.txt", " ")
    printf("Loaded data: %d %d\n", f.numRows(), f.numCols());
    
    val nystrom = new Nystrom();
    val resNystrom = nystrom.execute(sc, f, 8);
    printf("Result Nystrom: %d %d\n", resNystrom.numRows(), resNystrom.numCols());
    
    // testing lrlp
    val lp = new LabelPropagation();
    val f0 = Loader.readCoordinateMatrixFromTxt(sc, "data/f0.txt", " ")

    val resLP = lp.execute(sc, resNystrom.toBlockMatrix(128, 128), f0, 16, 0.1, 1e-9, 10);
    printf("Result LRLP: %d %d\n", resLP.numRows(), resLP.numCols());
    
  }
  
}
