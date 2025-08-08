import java.util.*;
import java.util.function.*;
//import java.util.stream.*;

public class KNearestNeighbors<C> implements Classifier<Float64Row, C> {
  /** Set of patterns collected till now. Usually modifiable and not thread-safe. */
  protected final List<Pair<Float64Row, C>> knownPatterns = new ArrayList<>();
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
    knownPatterns.add(new Pair<>(input, outputCls));
  }

  /** 
   * Select k smallest elements out of a sequence of elements. 
   * May return a sequence of size less than or equal to k.
   * This is a sequential operation only, parallel operation not allowed.
   */
  public static <T> Collection<T> selectK(Iterator<? extends T> sequence, Comparator<? super T> comparator, int k) {
    if (k <= 0) return Collections.emptyList();
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
    assert heap.size() <= k;
    return Collections.unmodifiableCollection(heap);
  }

  @Override public C predict(Float64Row input) {
    final var selected = selectK(knownPatterns.iterator(), (p1, p2) -> Double.compare(
       distanceFunction.applyAsDouble(p1.input(), input),
       distanceFunction.applyAsDouble(p2.input(), input)
    ), K);
    final var counts = new HashMap<C, Integer>();
    for(var p : selected) 
      counts.put(p.output(), counts.getOrDefault(p.output(), 0) + 1);
    if(counts.isEmpty()) throw new IllegalStateException("Nothing fitted in");
    C maxClass = null; int maxClassCount = 0;
    for(var entry : counts.entrySet()) {
      final int classCount = entry.getValue();
      if(classCount > maxClassCount) {
        maxClass = entry.getKey(); 
        maxClassCount = classCount;
      }
    }
    return maxClass;
  }
}
