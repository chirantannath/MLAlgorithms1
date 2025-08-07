import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class KNearestNeighbors<C> implements Classifier<Float64Row, C> {
  /** Combined type to store input-output combination. */
  protected static record Pattern<CLS> (Float64Row input, CLS outputCls) {}
  /** Set of patterns collected till now. Usually modifiable and not thread-safe. */
  protected final List<Pattern<C>> knownPatterns = new ArrayList<>();
  /** Parameter, number of nearest neighbors selected for prediction */
  public final int K;
  /** 
   * Parameter, distance function to be used, {@link Float64Row#distanceEuclidean(Float64Row)} 
   * by default.
   */
  public final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunction;

  public KNearestNeighbors(final int k, final ToDoubleBiFunction<Float64Row, Float64Row> distanceFunc) {
    if(k <= 0) throw new IllegalArgumentException();
    K = k;
    Objects.requireNonNull(distanceFunc);
    distanceFunction = distanceFunc;
  }
  public KNearestNeighbors(final int k) {
    this(k, Float64Row::distanceEuclidean);
  }

  @Override public void fit(Float64Row input, C outputCls) {
    knownPatterns.add(new Pattern<>(input, outputCls));
  }

  /** 
   * Select k smallest elements out of a sequence of elements. 
   * May return a sequence of size less than or equal to k.
   * This is a sequential operation only, parallel operation not allowed.
   */
  public static <T> Stream<T> selectK(Iterator<? extends T> sequence, Comparator<? super T> comparator, int k) {
    if (k <= 0) return Stream.empty();
    //I need the LARGEST elements first for this to work
    final var heap = new PriorityQueue<T>(comparator.reversed());
    while(sequence.hasNext()) {
      final T element = sequence.next();
      if(heap.size() < k) {
        heap.add(element); continue;
      }
      final T root = heap.peek();
      if(comparator.compare(element, root) >= 0) continue;
      //If element < root
      heap.poll(); heap.add(element);
    }
    return heap.stream();
  }

  @Override public C predict(Float64Row input) {
    return /*knownPatterns.stream()//.unordered().parallel()
    .sorted((p1, p2) -> Double.compare(
      distanceFunction.applyAsDouble(p1.input, input),
      distanceFunction.applyAsDouble(p2.input, input)
    ))
    .limit(K)//.parallel()*/
    selectK(knownPatterns.iterator(), (p1, p2) -> Double.compare(
      distanceFunction.applyAsDouble(p1.input, input),
      distanceFunction.applyAsDouble(p2.input, input)
    ), K)
    .collect(Collectors.groupingBy/*Concurrent*/(Pattern<C>::outputCls, Collectors.counting()))
    .entrySet().stream()//.unordered().parallel() //(K is assumed to be small)
    .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
    .orElseThrow(IllegalStateException::new)
    .getKey();
  }

  // public C predictParallel(Float64Row input) {
  //   return knownPatterns.parallelStream()/*.unordered()*/.parallel()
  //   .sorted((p1, p2) -> Double.compare(
  //     distanceFunction.applyAsDouble(p1.input, input),
  //     distanceFunction.applyAsDouble(p2.input, input)
  //   ))
  //   .limit(K).parallel()
  //   .collect(Collectors.groupingByConcurrent(Pattern<C>::outputCls, Collectors.counting()))
  //   .entrySet().stream()//.unordered().parallel() //(K is assumed to be small)
  //   .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
  //   .orElseThrow(IllegalStateException::new)
  //   .getKey();
  // }
}
