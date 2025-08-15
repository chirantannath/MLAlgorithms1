import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * K-medoid clustering.
 * 
 * @author chirantannath
 */
public final class KMedoidClusterer extends KMeansClusterer {
  public KMedoidClusterer(int k, ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc, Supplier<Random> rndSrc,
      long maxIter) {
    super(k, distanceFunc, rndSrc, maxIter);
  }

  public KMedoidClusterer(int k) {
    super(k);
  }

  @Override public Stream<Pair<Float64Row, ClusterLabel<Float64Row>>> apply(Supplier<Stream<Float64Row>> inputsSupplier) {
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
            .boxed()
            .collect(Collectors.groupingByConcurrent(dataIdx -> groupIDs[dataIdx],
                Collectors.mapping(dataIdx -> data[dataIdx], Float64RowStats.meanCollector(rowLength))));
        for (var entry : newGrpMap.entrySet())
          newGroupRepresentatives[entry.getKey()] = entry.getValue();
      }
      //In K-medoid, the means are replaced by the patterns closest to the means
      for(int grp = 0; grp < K; grp++) {
        final Float64Row grpMean = newGroupRepresentatives[grp];
        final Float64Row grpMedoid = Arrays.stream(data).unordered().parallel()
            .min((p1, p2) -> Double.compare(
              distanceFunction.applyAsDouble(p1, grpMean),
              distanceFunction.applyAsDouble(p2, grpMean)
            ))
            .orElseThrow(IllegalArgumentException::new);
        newGroupRepresentatives[grp] = grpMedoid;
      }
      if (Arrays.equals(groupRepresentatives, newGroupRepresentatives))
        break;
      System.arraycopy(newGroupRepresentatives, 0, groupRepresentatives, 0, K);
    }

    return IntStream.range(0, data.length).parallel()
        .mapToObj(dataIdx -> new Pair<>(data[dataIdx], new ClusterLabel<>(groupIDs[dataIdx], groupRepresentatives[groupIDs[dataIdx]])));
  }
}
