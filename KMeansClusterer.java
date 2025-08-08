import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * K-means clustering.
 * 
 * @author chirantannath
 */
public class KMeansClusterer implements Clusterer<Float64Row> {
  /** The number of groups to be output. */
  public final int K;
  /**
   * The distance function to be used, by default
   * {@link Float64Row#distanceEuclidean(Float64Row)}.
   */
  public final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction;
  /**
   * The random number generator source being used; by default
   * {@link java.util.Random#Random()}.
   */
  public final Supplier<Random> randomSource;
  /**
   * Maximum number of iterations, by default {@link java.lang.Short#MAX_VALUE} (=
   * 32767).
   */
  public final long maxIterations;

  public KMeansClusterer(int k, ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc, Supplier<Random> rndSrc,
      long maxIter) {
    if (k <= 0)
      throw new IllegalArgumentException("K");
    K = k;
    Objects.requireNonNull(distanceFunc, "distanceFunction");
    distanceFunction = distanceFunc;
    Objects.requireNonNull(rndSrc, "randomSource");
    randomSource = rndSrc;
    if (maxIter <= 0)
      throw new IllegalArgumentException("maxIterations");
    maxIterations = maxIter;
  }

  public KMeansClusterer(int k) {
    this(k, Float64Row::distanceEuclidean, Random::new, Short.MAX_VALUE);
  }

  @Override public Stream<Stream<Float64Row>> apply(Supplier<Stream<Float64Row>> inputsSupplier) {
    // Random source
    final Random rng = randomSource.get();
    Objects.requireNonNull(rng);
    // Extract data; we only need to do this once (since we're doing it anyway)
    final Float64Row[] data = inputsSupplier.get().unordered().parallel().toArray(Float64Row[]::new);
    final int rowLength = data[0].getRowLength();
    // Representatives of each group: here this is the mean
    final Float64Row[] groupRepresentatives;
    // First selection is K random elements out of data[]
    groupRepresentatives = rng.ints(0, K).unordered() // .parallel() //K is usually small
        .distinct().unordered() // .parallel()
        .limit(K).mapToObj(i -> data[i]).toArray(Float64Row[]::new);
    // Which data is in which group?
    final int[] groupIDs = new int[data.length];
    // Iterations
    for (long i = 0; i < maxIterations; i++) {
      // Compute new group IDs
      IntStream.range(0, data.length).unordered().parallel()
          .forEach(dataIdx -> {
            // Find to which group representative is this nearest to?
            // Search this sequentially (since K is small)
            final Float64Row element = data[dataIdx];
            int selectedGrp = -1;
            double distanceTo = Double.POSITIVE_INFINITY, temp;
            for (int grp = 0; grp < K; grp++) {
              temp = distanceFunction.applyAsDouble(element, groupRepresentatives[grp]);
              if (selectedGrp < 0 || temp < distanceTo) {
                selectedGrp = grp;
                distanceTo = temp;
              }
            }
            groupIDs[dataIdx] = selectedGrp;
          });

      // Recompute group representatives.
      final Float64Row[] newGroupRepresentatives = new Float64Row[K];
      Arrays.fill(newGroupRepresentatives, null);
      {
        final var newGrpMap = IntStream.range(0, data.length).unordered().parallel()
            .mapToObj(dataIdx -> new Pair<Float64Row, Integer>(data[dataIdx], groupIDs[dataIdx]))
            .collect(Collectors.groupingByConcurrent(Pair::output,
                Collectors.mapping(Pair::input, Float64RowStats.meanCollector(rowLength))));
        for (var entry : newGrpMap.entrySet())
          newGroupRepresentatives[entry.getKey()] = entry.getValue();
      }
      if (Arrays.equals(groupRepresentatives, newGroupRepresentatives))
        break;
      System.arraycopy(newGroupRepresentatives, 0, groupRepresentatives, 0, K);
    }

    // Now finally prepare groups
    final Map<Integer, Stream.Builder<Float64Row>> grpMap = new java.util.concurrent.ConcurrentHashMap<>();
    for (int grp = 0; grp < K; grp++)
      grpMap.put(grp, Stream.builder());
    IntStream.range(0, data.length).unordered().parallel()
        .forEach(dataIdx -> grpMap.get(groupIDs[dataIdx]).accept(data[dataIdx]));

    @SuppressWarnings("unchecked")
    final Stream<Float64Row>[] grps = (Stream<Float64Row>[]) new Stream<?>[K];
    for (var grpEntry : grpMap.entrySet())
      grps[grpEntry.getKey()] = grpEntry.getValue().build();
    return Arrays.stream(grps);
  }
}