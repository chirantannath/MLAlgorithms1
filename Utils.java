import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Miscellaneous functions not fitting into anything else.
 * 
 * @author chirantannath
 */
@SuppressWarnings("unused")
final class Utils {
  private Utils() {
  }

  private static record ConstantPredicate<T>(boolean logicValue) implements Predicate<T> {
    @Override
    public boolean test(T t) {
      return logicValue;
    }
  }

  private static final ConstantPredicate<?> CP_TRUE = new ConstantPredicate<>(true);
  private static final ConstantPredicate<?> CP_FALSE = new ConstantPredicate<>(false);

  @SuppressWarnings("unchecked")
  static <T> Predicate<T> constantPredicate(boolean logicValue) {
    return (Predicate<T>) (logicValue ? CP_TRUE : CP_FALSE);
  }

  /**
   * A set of integers internally storing using bit fields.
   * This set CANNOT STORE NEGATIVE VALUES. Also note that the
   * subset functions return an unmodifiable copy instead of a backing view on
   * the set.
   */
  public static final class WholeNumbersSet extends AbstractSet<Integer> implements SortedSet<Integer> {
    public final BitSet container;

    public WholeNumbersSet(BitSet container) {
      this.container = container;
    }

    public WholeNumbersSet() {
      this(new BitSet());
    }

    public WholeNumbersSet(int initialCapacity) {
      this(new BitSet(initialCapacity));
    }

    public WholeNumbersSet(Collection<Integer> c) {
      this(c.size());
      addAll(c);
    }

    @Override
    public int size() {
      return container.cardinality();
    }

    @Override
    public boolean isEmpty() {
      return container.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
      if (o instanceof Number n)
        return container.get(n.intValue());
      else
        return false;
    }

    @Override
    public Iterator<Integer> iterator() {
      return container.stream().parallel().boxed().iterator();
    }

    @Override
    public Object[] toArray() {
      return container.stream().parallel().boxed().toArray();
    }

    @Override
    public <T> T[] toArray(IntFunction<T[]> generator) {
      return container.stream().parallel().boxed().toArray(generator);
    }

    @Override
    public boolean add(Integer e) {
      final int i = e;
      final boolean wasPresent = container.get(i);
      container.set(i);
      return !wasPresent;
    }

    @Override
    public boolean remove(Object o) {
      if (o instanceof Number n) {
        final int i = n.intValue();
        final boolean wasPresent = container.get(i);
        container.clear(i);
        return wasPresent;
      } else
        return false;
    }

    @Override
    public void clear() {
      container.clear();
    }

    @Override
    public Spliterator<Integer> spliterator() {
      return container.stream().boxed().spliterator();
    }

    @Override
    public Stream<Integer> stream() {
      return container.stream().boxed();
    }

    @Override
    public Comparator<? super Integer> comparator() {
      return Comparator.naturalOrder();
    }

    @Override
    public Integer first() {
      if (isEmpty())
        throw new NoSuchElementException();
      return container.nextSetBit(0);
    }

    @Override
    public Integer last() {
      if (isEmpty())
        throw new NoSuchElementException();
      return container.length() - 1;
    }

    @Override
    public SortedSet<Integer> subSet(Integer fromElement, Integer toElement) {
      return Collections.unmodifiableSortedSet(new WholeNumbersSet(container.get(fromElement, toElement)));
    }

    @Override
    public SortedSet<Integer> headSet(Integer toElement) {
      return subSet(0, toElement);
    }

    @Override
    public SortedSet<Integer> tailSet(Integer fromElement) {
      return subSet(fromElement, container.length() - 1);
    }
  }

  @SuppressWarnings("unchecked")
  static <T> T[] concat(T[]... arrays) {
    if (arrays.length == 0)
      throw new IllegalArgumentException();
    int totalLength = 0, currentLength = arrays[0].length;
    final T[] result;
    for (T[] a : arrays)
      totalLength += a.length;
    result = Arrays.copyOf(arrays[0], totalLength);
    for (int i = 1; i < arrays.length; i++) {
      final T[] a = arrays[i];
      System.arraycopy(a, 0, result, currentLength, a.length);
      currentLength += a.length;
    }
    return result;
  }

  @SuppressWarnings("unchecked")
  static <T> T[] concatParallel(T[]... arrays) {
    if (arrays.length == 0)
      throw new IllegalArgumentException();
    final T[] prototype = Arrays.copyOf(arrays[0], 0);
    return Arrays.stream(arrays).parallel()
        .flatMap(Arrays::stream)
        .toArray(length -> Arrays.copyOf(prototype, length));
  }

  static int[] concat(int[]... arrays) {
    int totalLength = 0, currentLength = 0;
    final int[] result;
    for (int[] a : arrays)
      totalLength += a.length;
    result = new int[totalLength];
    for (int[] a : arrays) {
      System.arraycopy(a, 0, result, currentLength, a.length);
      currentLength += a.length;
    }
    assert currentLength == totalLength;
    return result;
  }

  static long[] concat(long[]... arrays) {
    int totalLength = 0, currentLength = 0;
    final long[] result;
    for (long[] a : arrays)
      totalLength += a.length;
    result = new long[totalLength];
    for (long[] a : arrays) {
      System.arraycopy(a, 0, result, currentLength, a.length);
      currentLength += a.length;
    }
    assert currentLength == totalLength;
    return result;
  }

  static double[] concat(double[]... arrays) {
    int totalLength = 0, currentLength = 0;
    final double[] result;
    for (double[] a : arrays)
      totalLength += a.length;
    result = new double[totalLength];
    for (double[] a : arrays) {
      System.arraycopy(a, 0, result, currentLength, a.length);
      currentLength += a.length;
    }
    assert currentLength == totalLength;
    return result;
  }

  static Stream<int[]> chooseRepeated(int k, int fromIndex, int toIndex) {
    final int N = toIndex - fromIndex;
    if (N < 0)
      throw new IllegalArgumentException();
    if (N == 0)
      return Stream.empty(); // Nothing to choose
    if (k >= N)
      return Stream.of(IntStream.range(fromIndex, toIndex).toArray()); // Choose all
    final var c1 = IntStream.range(fromIndex, toIndex).mapToObj(i -> new int[] { i });
    if (k == 1)
      return c1; // Choose 1
    return c1.flatMap(element -> chooseRepeated(k - 1, fromIndex, toIndex).map(element2 -> concat(element, element2)));
  }

  static Stream<int[]> chooseRepeatedParallel(int k, int fromIndex, int toIndex) {
    final int N = toIndex - fromIndex;
    if (N < 0)
      throw new IllegalArgumentException();
    if (N == 0)
      return Stream.empty(); // Nothing to choose
    if (k >= N)
      return Stream.of(IntStream.range(fromIndex, toIndex).parallel().toArray()); // Choose all
    final var c1 = IntStream.range(fromIndex, toIndex).parallel().mapToObj(i -> new int[] { i });
    if (k == 1)
      return c1; // Choose 1
    return c1.flatMap(
        element -> chooseRepeatedParallel(k - 1, fromIndex, toIndex).map(element2 -> concat(element, element2)));
  }

  static Stream<int[]> choose(int k, int fromIndex, int toIndex) {
    final int N = toIndex - fromIndex;
    if (N < 0)
      throw new IllegalArgumentException();
    if (N == 0)
      return Stream.empty(); // Nothing to choose
    if (k >= N)
      return Stream.of(IntStream.range(fromIndex, toIndex).toArray()); // Choose all
    final var c1 = IntStream.range(fromIndex, toIndex).mapToObj(i -> new int[] { i });
    if (k == 1)
      return c1; // Choose 1
    return c1.flatMap(element -> choose(k - 1, element[0] + 1, toIndex).map(element2 -> concat(element, element2)));
  }

  static Stream<int[]> chooseParallel(int k, int fromIndex, int toIndex) {
    final int N = toIndex - fromIndex;
    if (N < 0)
      throw new IllegalArgumentException();
    if (N == 0)
      return Stream.empty(); // Nothing to choose
    if (k >= N)
      return Stream.of(IntStream.range(fromIndex, toIndex).parallel().toArray()); // Choose all
    final var c1 = IntStream.range(fromIndex, toIndex).parallel().mapToObj(i -> new int[] { i });
    if (k == 1)
      return c1; // Choose 1
    return c1
        .flatMap(element -> chooseParallel(k - 1, element[0] + 1, toIndex).map(element2 -> concat(element, element2)));
  }

  static <T> Collection<T> selectSmallestK(Iterator<? extends T> sequence, Comparator<? super T> comparator, int k) {
    if (k <= 0)
      return Collections.emptyList();
    // I need the LARGEST elements first for this to work
    final var heap = new PriorityQueue<T>(comparator.reversed());
    while (sequence.hasNext()) {
      final T element = sequence.next();
      if (heap.size() < k) {
        heap.add(element);
        continue;
      }
      final T root = heap.peek();
      if (comparator.compare(element, root) >= 0)
        continue;
      // If element < root
      heap.poll();
      heap.add(element);
    }
    assert heap.size() <= k;
    return Collections.unmodifiableCollection(heap);
  }

  /** Substitute for std::lower_bound in C++. */
  static int lowerBoundIndex(int key, int[] arr) {
    int l = 0, h = arr.length, mid;
    while (l < h) {
      mid = (l + h) >>> 1;
      if (arr[mid] < key)
        l = mid + 1;
      else
        h = mid;
    }
    return l;
  }

  static <T> int lowerBoundIndex(T key, IntFunction<? extends T> arr, int startInclusive, int endExclusive,
      Comparator<? super T> comparator) {
    int l = startInclusive, h = endExclusive, mid;
    while (l < h) {
      mid = (l + h) >>> 1;
      if (comparator.compare(arr.apply(mid), key) < 0)
        l = mid + 1;
      else
        h = mid;
    }
    return l;
  }

  /** Substitute for std::higher_bound in C++. */
  static int higherBoundIndex(int key, int[] arr) {
    int l = 0, h = arr.length, mid;
    while (l < h) {
      mid = (l + h) >>> 1;
      if (key >= arr[mid])
        l = mid + 1;
      else
        h = mid;
    }
    return l;
  }

  static <T> int higherBoundIndex(T key, IntFunction<? extends T> arr, int startInclusive, int endExclusive,
      Comparator<? super T> comparator) {
    int l = startInclusive, h = endExclusive, mid;
    while (l < h) {
      mid = (l + h) >>> 1;
      if (comparator.compare(key, arr.apply(mid)) >= 0)
        l = mid + 1;
      else
        h = mid;
    }
    return l;
  }

  static int[] cumulativeSum(int... array) {
    final int[] result = Arrays.copyOf(array, array.length);
    for (int i = 1; i < array.length; i++)
      result[i] += result[i - 1];
    return result;
  }

  static long[] cumulativeSum(long... array) {
    final long[] result = Arrays.copyOf(array, array.length);
    for (int i = 1; i < array.length; i++)
      result[i] += result[i - 1];
    return result;
  }

  static double[] cumulativeSum(double... array) {
    final double[] result = Arrays.copyOf(array, array.length);
    for (int i = 1; i < array.length; i++)
      result[i] += result[i - 1];
    return result;
  }

  /** Returns the first quartile, assumes sorted array. */
  static double firstQuartile(double[] array, int startInclusive, int endExclusive) {
    final int N = endExclusive - startInclusive;
    return N == 1 ? array[startInclusive] : Utils.median(array, startInclusive, startInclusive + (N >>> 1));
  }

  /** Returns the third quartile, assumes sorted array. */
  static double thirdQuartile(double[] array, int startInclusive, int endExclusive) {
    final int N = endExclusive - startInclusive;
    return N == 1 ? array[startInclusive] : Utils.median(array, startInclusive + ((N + 1) >>> 1), endExclusive);
  }

  /** Returns the median, assumes sorted array. */
  static double median(double[] array, int startInclusive, int endExclusive) {
    Objects.checkFromToIndex(startInclusive, endExclusive, array.length);
    final int N = endExclusive - startInclusive;
    if (N <= 0)
      throw new IllegalArgumentException();
    final int midIdx = startInclusive + (N >>> 1);
    final double midValue2 = array[midIdx];
    return (N & 1) == 0 ? Math.scalb(array[midIdx - 1] + midValue2, -1) : midValue2;
  }

  /** Natural logarithm of 2. */
  static final double LOG_OF_2 = Math.log(2);

  /** Returns logarithm base 2 of {@code x}. */
  static double log2(double x) {
    return Math.log(x) / LOG_OF_2;
  }

  /** Counts (distinct) objects. Just a common wrapper. */
  static <T> Map<T, Long> valueCounts(final Stream<T> objects) {
    return Collections.unmodifiableMap(objects.unordered().parallel()
        .collect(Collectors.groupingByConcurrent(Function.identity(), Collectors.counting())));
  }

  /** Assumes elements are of type long. */
  static double countedEntropy(final Supplier<Stream<? extends Number>> counts) {
    final long total = counts.get().unordered().mapToLong(Number::longValue).sum();
    return counts.get().unordered().mapToDouble(x -> {
      final var probability = x.doubleValue() / total;
      return -1 * probability * log2(probability);
    }).sum();
  }

  /** Assumes elements are of type double. */
  static double weightedEntropy(final Supplier<Stream<? extends Number>> weights) {
    final double total = weights.get().unordered().mapToDouble(Number::doubleValue).sum();
    return weights.get().unordered().mapToDouble(x -> {
      final var probability = x.doubleValue() / total;
      return -1 * probability * log2(probability);
    }).sum();
  }

  /** Counts (distinct) objects. and computes entropy. */
  static double countedEntropy(final Stream<?> objects) {
    final var counts = valueCounts(objects);
    if (counts.size() <= 1)
      return 0D; // Mathematically proven
    return countedEntropy(counts.values()::stream);
  }

  static double countedGiniImpurity(final Supplier<Stream<? extends Number>> counts) {
    final long total = counts.get().unordered().mapToLong(Number::longValue).sum();
    return 1D - counts.get().unordered().mapToDouble(x -> {
      final var probability = x.doubleValue() / total;
      return probability * probability;
    }).sum();
  }

  static double weightedGiniImpurity(final Supplier<Stream<? extends Number>> weights) {
    final double total = weights.get().unordered().mapToDouble(Number::doubleValue).sum();
    return 1D - weights.get().unordered().mapToDouble(x -> {
      final var probability = x.doubleValue() / total;
      return probability * probability;
    }).sum();
  }

  static double countedGiniImpurity(final Stream<?> objects) {
    final var counts = valueCounts(objects);
    if (counts.size() <= 1)
      return 0D; // Mathematically proven
    return countedGiniImpurity(counts.values()::stream);
  }
}
