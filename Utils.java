import java.util.*;
import java.util.stream.*;
import java.util.function.*;

/**
 * Miscellaneous functions not fitting into anything else.
 * 
 * @author chirantannath
 */
@SuppressWarnings("unused")
final class Utils {
  private Utils() {
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

  static <T> Collection<T> selectK(Iterator<? extends T> sequence, Comparator<? super T> comparator, int k) {
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

}
