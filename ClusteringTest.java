import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.*;
import java.util.function.*;
import java.util.stream.*;

final class ClusteringTest {
  private ClusteringTest() {
  }

  public static void main(String args[]) throws Exception {
    // Assumes a classification dataset, discards output classification column
    final Float64Dataset input;
    final String[] columnNames;
    final int[] inFeatures;

    try (Scanner sc = new Scanner(System.in)) {
      {
        System.out.print("Enter dataset CSV path: ");
        String path = sc.nextLine();
        System.out.println("Reading...");
        var original = StringDataset.readCSV(path);
        System.out.println("Number of rows: " + original.size());
        System.out.println("Columns: " + Arrays.toString(original.getColumnNames()));
        System.out.println("Discarding last column");
        inFeatures = IntStream.range(0, original.getNumColumns() - 1).toArray();
        original = original.project(inFeatures);
        input = new Float64Dataset(original);
        columnNames = input.getColumnNames();
        System.out.println("Final columns: " + Arrays.toString(columnNames));
      }

      System.out.print("Use IQR scaling?[y/n]: ");
      final boolean useRobust = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
      if (useRobust)
        Float64Dataset.inplaceFitAndTransform(input, new RobustScaler(inFeatures.length));
      System.out.print("Use standardization?[y/n]: ");
      final boolean useStandardization = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
      if (useStandardization)
        Float64Dataset.inplaceFitAndTransform(input, new StandardScaler(inFeatures.length));

      System.out.println();

      System.out.print("Enter number of groups (K): ");
      final int K = Integer.parseInt(sc.nextLine().trim());
      System.out.print("Use randomized seed?[y/n]: ");
      final var useRandomizedSeed = Character.toLowerCase(sc.nextLine().trim().charAt(0)) == 'y';
      final Supplier<Random> rndSrc = useRandomizedSeed ? Random::new
          : () -> new Random(0x12012001_12012001L);
      final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc = Float64Row::distanceEuclidean;
      Clusterer<Float64Row> clusterer = new KMeansClusterer(K, distanceFunc, rndSrc, 0xFFFFL);
      testClustering(clusterer, distanceFunc, input, K, inFeatures.length);

      System.out.println();
      System.gc();

      clusterer = new KMedoidClusterer(K, distanceFunc, rndSrc, 0xFFFFL);
      testClustering(clusterer, distanceFunc, input, K, inFeatures.length);
    }
  }

  @SuppressWarnings("unchecked")
  static void testClustering(
      final Clusterer<Float64Row> clusterer,
      final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction,
      final List<Float64Row> input,
      final int numGroups,
      final int rowLength) {
    long ts;
    System.out.println(clusterer.getClass().getName() + " clustering starting...");
    ts = System.currentTimeMillis();
    List<Pair<Float64Row, ClusterLabel<Float64Row>>> result = clusterer
        .apply(input::parallelStream)
        .collect(Collectors.toUnmodifiableList());
    ts = System.currentTimeMillis() - ts;
    System.out.println("Cluster executed for " + ts + "ms");

    // Count and remove those that couldn't be classified.
    final AtomicLong ungroupedCount = new AtomicLong(0L);
    result = result.parallelStream().filter(p -> {
      final boolean empty = p.output().isEmpty();
      if (empty)
        ungroupedCount.incrementAndGet();
      return !empty;
    }).collect(Collectors.toUnmodifiableList());
    System.out.println("Unclassified rows: " + ungroupedCount);

    // Cluster information in arrays
    @SuppressWarnings("unused")
    final ClusterLabel<Float64Row>[] clusterArray;
    final List<Float64Row>[] clusterContents;
    final Float64Row[] centroids;

    {
      // Create groups
      Map<ClusterLabel<Float64Row>, List<Float64Row>> groups = result.parallelStream().unordered()
          .collect(Collectors.groupingByConcurrent(Pair::output,
              Collectors.mapping(Pair::input, Collectors.toUnmodifiableList())));

      // Set representative for those that do not have a representative (by mean)
      // And calculate centroids anyway
      final Map<ClusterLabel<Float64Row>, Float64Row> centroidsMap = new ConcurrentHashMap<>();
      // Centroids map is thread-safe
      {
        final AtomicBoolean changed = new AtomicBoolean(false);
        final var g2a = (Map.Entry<ClusterLabel<Float64Row>, List<Float64Row>>[]) groups.entrySet().parallelStream()
            .map(entry -> {
              var cluster = entry.getKey();
              final var mean = entry.getValue().parallelStream().unordered()
                  .collect(Float64RowStats.meanCollector(rowLength));

              if (cluster.hasRepresentative()) {
                centroidsMap.put(cluster, mean);
                return entry;
              }
              changed.set(true);
              cluster = new ClusterLabel<>(cluster.label(), mean);
              centroidsMap.put(cluster, mean);
              return Map.entry(cluster, entry.getValue());
            }).toArray(Map.Entry<?, ?>[]::new);
        if (changed.get()) {
          groups = Map.ofEntries(g2a);
          // Need to update result as well
          result = Arrays.stream(g2a).parallel().flatMap(e -> e.getValue().parallelStream()
              .map(row -> new Pair<Float64Row, ClusterLabel<Float64Row>>(row, e.getKey())))
              .collect(Collectors.toUnmodifiableList());
        }
      }

      assert numGroups == groups.size();

      // Array of cluster labels
      // No need to parallelize here
      clusterArray = groups.keySet()
          .toArray((ClusterLabel<Float64Row>[]) new ClusterLabel<?>[numGroups]);
      clusterContents = groups.values()
          .toArray((List<Float64Row>[]) new List<?>[numGroups]);
      centroids = groups.keySet().stream().map(centroidsMap::get).toArray(Float64Row[]::new);
    }

    // Calculate distortion measure
    final var distortionMeasure = result.parallelStream().unordered()
        .mapToDouble(pair -> distanceFunction.applyAsDouble(pair.input(),
            pair.output().representative()))
        .summaryStatistics();
    System.out.println("Distortion measure statistics: " + distortionMeasure);

    // Calculate (total) intra-cluster distance, similar to distortion measure.
    final var intraClusterDist = IntStream.range(0, numGroups)
        .unordered().parallel()
        .mapToDouble(grp -> clusterContents[grp].parallelStream().unordered()
            .mapToDouble(row -> distanceFunction.applyAsDouble(row, centroids[grp])).sum())
        .summaryStatistics();
    System.out.println("Intra-cluster distance statistics (calculated sum per cluster): " + intraClusterDist);

    // Centroid inter-cluster distance statistics
    final var interClusterDist = Utils.chooseParallel(2, 0, numGroups)
        .unordered().parallel()
        .mapToDouble(
            p -> distanceFunction.applyAsDouble(centroids[p[0]], centroids[p[1]]))
        .summaryStatistics();
    System.out.println("Inter-cluster centroid distance statistics: " + interClusterDist);

    // Minimum distance between 2 points in 2 different clusters
    final var singleLinkageDist = Utils.chooseParallel(2, 0, numGroups)
        .unordered().parallel()
        .map(p -> clusterContents[p[0]].parallelStream().unordered().flatMapToDouble(
            r1 -> clusterContents[p[1]].parallelStream().unordered()
                .mapToDouble(r2 -> distanceFunction.applyAsDouble(r1, r2)))
            .min())
        // .filter(OptionalDouble::isPresent) //Is this required?
        .mapToDouble(OptionalDouble::orElseThrow)
        .summaryStatistics();
    System.out.println("Single-linkage distance per 2 clusters stats: " + singleLinkageDist);

    // Maximum distance between 2 points in 2 different clusters
    final var completeLinkageDist = Utils.chooseParallel(2, 0, numGroups)
        .unordered().parallel()
        .map(p -> clusterContents[p[0]].parallelStream().unordered().flatMapToDouble(
            r1 -> clusterContents[p[1]].parallelStream().unordered()
                .mapToDouble(r2 -> distanceFunction.applyAsDouble(r1, r2)))
            .max())
        // .filter(OptionalDouble::isPresent) //Is this required?
        .mapToDouble(OptionalDouble::orElseThrow)
        .summaryStatistics();
    System.out.println("Complete-linkage distance per 2 clusters stats: " + completeLinkageDist);
  }
}
