import java.util.*;
import java.util.stream.*;
import java.util.function.*;

final class ClusteringTest {
  private ClusteringTest() {}
  
  public static void main(String args[]) throws Exception {
    //Assumes a classification dataset, discards output classification column
    final Float64Dataset input;
    final String[] columnNames;
    final int[] inFeatures;
    
    System.out.println("Reading...");
    {
      var original = StringDataset.readCSV(args.length > 0 ? args[0] : "diabetes.csv");
      System.out.println("Number of rows: "+original.size());
      System.out.println("Columns: "+Arrays.toString(original.getColumnNames()));
      System.out.println("Discarding last column");
      inFeatures = IntStream.range(0, original.getNumColumns()-1).toArray();
      original = original.project(inFeatures);
      input = new Float64Dataset(original);
      columnNames = input.getColumnNames();
      System.out.println("Final columns: "+Arrays.toString(columnNames));
    }
    
    System.out.println();
    
    final int K = args.length > 1 ? Integer.parseInt(args[1]) : 2;
    final Supplier<Random> rndSrc = Arrays.asList(args).contains("rnd") ? 
        Random::new : () -> new Random(0x12012001_12012001L);
    final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc = Float64Row::distanceEuclidean;
    Clusterer<Float64Row> clusterer = new KMeansClusterer(K, distanceFunc, rndSrc, 0xFFFFL);
    testClustering(clusterer, distanceFunc, input, K);
    
    System.out.println();
    System.gc();
    
    clusterer = new KMedoidClusterer(K, distanceFunc, rndSrc, 0xFFFFL);
    testClustering(clusterer, distanceFunc, input, K);
  }
  
  static void testClustering(
      final Clusterer<Float64Row> clusterer,
      final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction, 
      final List<Float64Row> input, 
      final long numGroups) {
    long ts;
    System.out.println(clusterer.getClass().getName()+" clustering starting...");
    ts = System.currentTimeMillis();
    final List<Pair<Float64Row, ClusterLabel<Float64Row>>> result = clusterer
        .apply(input::parallelStream)
        .collect(Collectors.toUnmodifiableList());
    ts = System.currentTimeMillis() - ts;
    System.out.println("Cluster executed for "+ts+"ms");
    
    //Calculate distortion measure
    final var distortionMeasure = result.parallelStream().unordered()
    .mapToDouble(pair -> distanceFunction.applyAsDouble(pair.input(), pair.output().representative()))
    .summaryStatistics();
    System.out.println("Distortion measure statistics: "+distortionMeasure);
    
    //Create groups
    final Map<ClusterLabel<Float64Row>, List<Float64Row>> groups = result.parallelStream().unordered()
    .collect(Collectors.groupingByConcurrent(Pair::output, Collectors.mapping(Pair::input, Collectors.toUnmodifiableList())));
     
  }
}
