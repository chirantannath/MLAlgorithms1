import java.util.*;
import java.util.stream.*;

/**
 * Miscellaneous functions not fitting into anything else.
 * 
 * @author chirantannath
 */
final class Utils {
  private Utils() {
  }

  static <T> T[] concat(T[] a1, T[] a2) {
    final T[] result = Arrays.copyOf(a1, a1.length + a2.length);
    System.arraycopy(a2, 0, result, a1.length, a2.length);
    return result;
  }

  static int[] concat(int[] a1, int[] a2) {
    final int[] result = Arrays.copyOf(a1, a1.length + a2.length);
    System.arraycopy(a2, 0, result, a1.length, a2.length);
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
    return c1.flatMap(element -> chooseRepeatedParallel(k - 1, fromIndex, toIndex).map(element2 -> concat(element, element2)));
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
    return c1.flatMap(element -> choose(k - 1, element[0]+1, toIndex).map(element2 -> concat(element, element2)));
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
    return c1.flatMap(element -> chooseParallel(k - 1, element[0]+1, toIndex).map(element2 -> concat(element, element2)));
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

}
