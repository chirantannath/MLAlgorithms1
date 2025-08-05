import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.*;

/**
 * Represents a table for data.
 * @param <R> the (sub)type for a single row
 * @author chirantannath
 */
public class Dataset<R extends Row> implements java.util.function.IntFunction<R> {
  /** Column names. */
  private final String[] columnNames;
  /** Row type. */
  public final Class<R> rowType;
  /** Data stored in <strong>row major</strong> order. */
  private final R[] data;
  
  public Dataset(String[] columnNames, Class<R> rowType, R[] data) throws IllegalArgumentException, NullPointerException {
    Objects.requireNonNull(columnNames, "columnNames");
    Objects.requireNonNull(rowType, "rowType");
    Objects.requireNonNull(data, "data");
    this.columnNames = Arrays.copyOf(columnNames, columnNames.length);
    this.rowType = rowType;
    //Proper column length for all?
    for(int i = 0; i < data.length; i++)
      if(data[i].getRowLength() != columnNames.length)
        throw new IllegalArgumentException("column number mismatch at position "+i);
    this.data = Arrays.copyOf(data, data.length);
  }
  @SuppressWarnings("unchecked") public Dataset(String[] columnNames, Class<R> rowType, Stream<R> data) throws IllegalArgumentException, NullPointerException {
    this(columnNames, rowType, data.toArray(len -> (R[])Array.newInstance(rowType, len)));
  }
  @SuppressWarnings("unchecked") public Dataset(Dataset<? extends R> other) {
    this(other.columnNames, (Class<R>)other.rowType, other.data);
  }

  public int getNumColumns() {return columnNames.length;}
  public String getColumnName(int index) throws IndexOutOfBoundsException {return columnNames[index];}
  public String[] getColumnNames() {return Arrays.copyOf(columnNames, columnNames.length);}
  
  public int getNumDataRows() {return data.length;}
  public R getDataRow(int index) throws IndexOutOfBoundsException {return data[index];}
  public void setDataRow(int index, R value) throws IndexOutOfBoundsException, IllegalArgumentException, NullPointerException {
    if(value.getRowLength() != columnNames.length)
      throw new IllegalArgumentException();
    data[index] = value;
  }
  public synchronized void setSynchronized(int index, R value) throws IndexOutOfBoundsException, IllegalArgumentException, NullPointerException {
    setDataRow(index, value);
  }
  @Override public R apply(int index) {return getDataRow(index);}
  public R[] getDataRows() {return Arrays.copyOf(data, data.length);}
  public Stream<R> stream() {return Arrays.stream(data);}
  public Stream<?> columnStream(int column) throws IndexOutOfBoundsException {
    Objects.checkIndex(column, getNumColumns());
    return stream().map(row -> row.get(column));
  }

  public Dataset<?> project(final int ... indices) throws IndexOutOfBoundsException {
    return new Dataset<>(
      Arrays.stream(indices).mapToObj(this::getColumnName).toArray(String[]::new),
      Row.class,
      stream().parallel().map(row -> row.project(indices))
    );
  }
}
