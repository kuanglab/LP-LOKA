# Protein Remote Homology Detection and Fold Recognition in Massive Protein Networks

## Serial MATLAB / Octave Version

### Requirements
- MATLAB 2013a
- or Octave 4.0.0

### Examples

- Nyström Approximation:

```
S = rand(16,4); % <-- your data here
F = LPLOKA_Nystrom(S);
```

- Low-rank label propagation

```
f_hat = LPLOKA(F, f0);
```

Optional parameters:

* **Alpha**: label propagation hyper-parameter (default 0.1)
* **MaxIter**: maximum number of iterations (default 20)
* **Tol**: convergence tolerance (default 1e-9)

Example:

```
f_hat = LPLOKA(F, f0, 'Alpha', 0.9, 'MaxIter', 20, 'Tol', 1e-8);
```

## Parallel C++ MPI/OpenMP Version

### Software Requirements

* Open MPI 1.10.2
* MacOS 10.12.6 or Linux Ubuntu 16.04

### Compilation
You can compile the project using `make` command. To compile individual tools, you can use:

* Nyström approximation:

```
make nystrom
```

* Low-Rank Label Propagation: 

```
make lploka
```

### Examples

#### Nyström approximation:

```
mpirun -np <processes> ./build/nystrom -i <input_path> -n <rows> -k <columns> --verbose -o <output_path> -p <precision>
```

The list of accepted parameters are the following

* **-np (required)**: number of processes to be used by MPI
* **-i (required)**: path to the input data. Data should have elements separated by a "space" character, and each row should end with a breakline (\n) character
* **-n (required)**: number of rows in the input data
* **-k (required)**: number of columns in the input data
* **--verbose (optional)**: show information while running the algorithm
* **-o (optional)**: path to the output files. If not provided, output will not be saved. It will be saved one file per process, such as `<output_path>1.txt` and `<output_path>2.txt` if `np` is 2
* **-p (optional)**: precision to write output (default 6 decimal places)

Example:

```
mpirun -np 2 ./build/nystrom -i data/input.txt -n 16 -k 5 --verbose -o data/output_ -p 16
```

#### Low-rank label propagation:

```
mpirun -np <processes> ./build/lploka -F <input_path> -f0 <input_f0> -n <rows> -k <low_rank> -startIndex <start> -endIndex <end> -maxIter <max_iter> -tol <tolerance> --verbose -alpha <alpha> -saveRows <num_rows> -o <output_path> -p <precision>
```

The list of accepted parameters are the following

* **-np (required)**: number of processes to be used by MPI
* **-F (required)**: path to the input data. Data should have elements separated by a "space" character, and each row should end with a breakline (\n) character
* **-f0 (required)**: path to where the f0 vectors are located. Files should be named: `<input_f0>/1.txt`, `<input_f0>/2.txt`, and so on. In each file, each row should end with a breakline (\n) character
* **-n (required)**: number of rows in the input data
* **-k (required)**: number of columns in the input data
* **-startIndex (optional)**: first file index to use as f0. For example, if `startIndex = 0`, then the first `f0` read will be `<input_f0>/1.txt` (default 0)
* **-endIndex (optional)**: last file index to use as `f0`. For example, if `endIndex = 5`, then the last `f0` read will be `<input_f0>/5.txt` (default 0)
* **-o (optional)**: path to the output files. If not provided, output will not be saved. It will be saved one file per `startIndex ` to `endIndex `, such as `<output_path>/1.txt` and `<output_path>/2.txt` if `startIndex = 0`, `endIndex = 1`
* **-maxIter (optional)**: maximum number of iterations (default 1000)
* **-tol (optional)**: convergence tolerance (default 1e-9)
* **--verbose (optional)**: show information while running the algorithm
* **-alpha (optional)**: label propagation hyper-parameter (default 0.1)
* **-saveRows (optional)**: number of rows of `f` to save (default 0)
* **-p (optional)**: precision to write output (default 6 decimal places)

Example:

```
mpirun -np <processes> ./build/lploka -F data/F.txt -f0 data/f0.txt -n 16 -k 5 -startIndex 0 -endIndex 5 -maxIter 20 -tol 1e9 --verbose -alpha 0.9 -saveRows 13710 -o output -p 6
```

## Parallel Scala Hadoop/Spark Version

### Software Requirements

* Scala 2.11
* Spark Project Core 1.6.3
* Spark Project ML Library 1.6.3
* Maven 2


### Examples

- Nyström Approximation:

```
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import edu.umn.cs.kuanglab.lploka.tools.Loader
import edu.umn.cs.kuanglab.lploka.Nystrom
import edu.umn.cs.kuanglab.lploka.LabelPropagation

// testing Nystrom
val f = Loader.readCoordinateMatrixFromTxt(sc, "data/X.txt", " ")
val nystrom = new Nystrom();
val resNystrom = nystrom.execute(sc, f, 8);
```

- Low-rank label propagation

```
val lp = new LabelPropagation();
val f0 = Loader.readCoordinateMatrixFromTxt(sc, "data/f0.txt", " ")

val resLP = lp.execute(sc, resNystrom.toBlockMatrix(128, 128), f0, 16, 0.1, 1e-9, 10);
```
