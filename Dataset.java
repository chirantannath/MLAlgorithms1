import java.lang.reflect.Array;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Represents a table for data.
 * 
 * @param <R> the (sub)type for a single row
 * @author chirantannath
 */
public class Dataset<R extends Row> extends AbstractList<R> implements RandomAccess, java.util.function.IntFunction<R> {
  /** Column names. */
  private final String[] columnNames;
  /** Row type. */
  public final Class<R> rowType;
  /** Data stored in <strong>row major</strong> order. */
  private final R[] data;

  public Dataset(String[] columnNames, Class<R> rowType, R[] data)
      throws IllegalArgumentException, NullPointerException {
    Objects.requireNonNull(columnNames, "columnNames");
    Objects.requireNonNull(rowType, "rowType");
    Objects.requireNonNull(data, "data");
    this.columnNames = Arrays.copyOf(columnNames, columnNames.length);
    this.rowType = rowType;
    // Proper column length for all?
    for (int i = 0; i < data.length; i++)
      if (data[i].getRowLength() != columnNames.length)
        throw new IllegalArgumentException("column number mismatch at position " + i);
    this.data = Arrays.copyOf(data, data.length);
  }

  @SuppressWarnings("unchecked")
  public Dataset(String[] columnNames, Class<R> rowType, Stream<? extends R> data)
      throws IllegalArgumentException, NullPointerException {
    this(columnNames, rowType, data.toArray(len -> (R[]) Array.newInstance(rowType, len)));
  }

  @SuppressWarnings("unchecked")
  public Dataset(Dataset<? extends R> other) {
    columnNames = Arrays.copyOf(other.columnNames, other.columnNames.length);
    rowType = (Class<R>) other.rowType;
    data = Arrays.copyOf(other.data, other.data.length);
  }

  public int getNumColumns() {
    return columnNames.length;
  }

  public String getColumnName(int index) throws IndexOutOfBoundsException {
    return columnNames[index];
  }

  public String[] getColumnNames() {
    return Arrays.copyOf(columnNames, columnNames.length);
  }

  public int getNumDataRows() {
    return data.length;
  }

  public R getDataRow(int index) throws IndexOutOfBoundsException {
    return data[index];
  }

  public void setDataRow(int index, R value)
      throws IndexOutOfBoundsException, IllegalArgumentException, NullPointerException {
    if (value.getRowLength() != columnNames.length)
      throw new IllegalArgumentException();
    data[index] = value;
  }

  public synchronized void setSynchronized(int index, R value)
      throws IndexOutOfBoundsException, IllegalArgumentException, NullPointerException {
    setDataRow(index, value);
  }

  @Override
  public R apply(int index) {
    return getDataRow(index);
  }

  public R[] getDataRows() {
    return Arrays.copyOf(data, data.length);
  }

  public Stream<?> columnStream(int column) throws IndexOutOfBoundsException {
    Objects.checkIndex(column, getNumColumns());
    return stream().map(row -> row.get(column));
  }

  public Dataset<?> project(final int... indices) throws IndexOutOfBoundsException {
    return new Dataset<>(
        Arrays.stream(indices).mapToObj(this::getColumnName).toArray(String[]::new),
        Row.class,
        stream().parallel().map(row -> row.project(indices)));
  }

  @Override
  public int size() {
    return getNumDataRows();
  }

  @Override
  public R get(int index) {
    return getDataRow(index);
  }

  @Override
  public R set(int index, R row) {
    final R old = data[index];
    data[index] = row;
    return old;
  }

  @Override
  public Dataset<R> subList(int fromIndex, int toIndex) {
    Objects.checkFromToIndex(fromIndex, toIndex, getNumDataRows());
    final var stream = IntStream.range(fromIndex, toIndex).parallel().mapToObj(this::getDataRow);
    return new Dataset<>(columnNames, rowType, stream);
  }

  @Override
  public Spliterator<R> spliterator() {
    return Arrays.spliterator(data);
  }

  /** (Lazy) split ({@code R} doesn't even have to be a row) */
  @SuppressWarnings("unchecked")
  public static <R> Stream<Stream<R>> split(final List<? extends R> source, Random rng, double... partitions) {
    if (partitions.length == 0)
      return Stream.of((Stream<R>) source.stream());
    final int totalSize = source.size();

    final int numPartitions = partitions.length + 1;
    final int[] partitionLengths = new int[numPartitions];
    {
      int currentAllocated = 0;
      for (int i = 0; i < partitions.length; i++) {
        if (partitions[i] < 0 || partitions[i] > 1)
          throw new IllegalArgumentException("partition number " + i);
        final int partitionSize = (int) (totalSize * partitions[i]);
        final int nextPartitionStart = currentAllocated + partitionSize;
        if (partitionSize < 0 || nextPartitionStart > totalSize)
          throw new IllegalArgumentException("partition number " + i);
        partitionLengths[i] = partitionSize;
        currentAllocated = nextPartitionStart;
      }
      partitionLengths[numPartitions - 1] = totalSize - currentAllocated;
    }

    final int[] indices = IntStream.range(0, totalSize).toArray();
    if (rng != null) {
      // Shuffling algorithm adapated from the default implementation in OpenJDK 17
      int t, j;
      for (int i = 0; i < totalSize - 1; i++) {
        j = rng.nextInt(i, totalSize);
        t = indices[i];
        indices[i] = indices[j];
        indices[j] = t;
      }
    }

    // NOT unordered here. If rng is seeded, needs to be deterministic.
    return IntStream.range(0, numPartitions).parallel()
        .mapToObj(partitionIndex -> {
          final int partitionStart = (partitionIndex == 0) ? 0 : partitionLengths[partitionIndex - 1];
          final int partitionEnd = partitionStart + partitionLengths[partitionIndex];
          final Stream<R> selected = IntStream.range(partitionStart, partitionEnd).parallel()
              .mapToObj(i -> source.get(indices[i]));
          return selected;
        });
  }

  /**
   * Same as {@link #split(List<? extends R>, Random, double...)}, splits the
   * input source sequence into {@code K} groups lazily.
   */
  @SuppressWarnings("unchecked")
  public static <R> Stream<Stream<R>> kFold(final List<? extends R> source, final int K) {
    if (K <= 1)
      return Stream.of((Stream<R>) source.stream());
    final int totalSize = source.size();
    final int[] partitionLengths = new int[K];
    {
      final int rawPartitionSize = totalSize / K;
      int elementsLeft = totalSize % K; // Those we couldn't place into blocks directly
      Arrays.fill(partitionLengths, rawPartitionSize);
      for (int i = 0; i < K && elementsLeft > 0; i++) {
        // Distribute among some of the blocks, starting from first.
        partitionLengths[i]++;
        elementsLeft--;
      }
    }

    // No need to shuffle.
    return IntStream.range(0, K).parallel()
        .mapToObj(partitionIndex -> {
          final int partitionStart = (partitionIndex == 0) ? 0 : partitionLengths[partitionIndex - 1];
          final int partitionEnd = partitionStart + partitionLengths[partitionIndex];
          final Stream<R> selected = IntStream.range(partitionStart, partitionEnd).parallel()
              .mapToObj(source::get);
          return selected;
        });
  }

  /**
   * Similar to {@link #kFold(List<? extends R>, int)}, but selects one block each
   * time as the smaller block, merges the other blocks into a bigger block, and
   * calls {@code action.accept()} {@code K} times. The smaller block is sent as
   * the <i>second</i> argument.
   */
  @SuppressWarnings("unchecked")
  public static <R> void doKFold(final List<? extends R> source, final int K,
      BiConsumer<? super Stream<? super R>, ? super Stream<? super R>> action) {
    if (K <= 1)
      return;
    Objects.requireNonNull(action, "action");
    final Stream<R>[] blocks = (Stream<R>[]) kFold(source, K).toArray(Stream<?>[]::new);

    for (int i = 0; i < K; i++) {
      final int current = i;
      final Stream<R> bigBlock = IntStream.range(0, K).parallel()
          .filter(blockIdx -> blockIdx != current)
          .mapToObj(blockIdx -> blocks[blockIdx].parallel()).flatMap(Function.identity());
      action.accept(bigBlock, blocks[current]);
    }
  }
}
