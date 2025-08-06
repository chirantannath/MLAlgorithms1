import java.math.BigDecimal;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;
/**
 * Default implementation of {@link Row} featuring only {@link java.lang.String} type attributes,
 * some or all of which may be {@code null}. After creation we can change attributes but not 
 * their number.
 * @author chirantannath
 */
public final class StringRow implements Row, IntFunction<String>{
  private final String[] data;

  public StringRow(String[] data) {
    Objects.requireNonNull(data);
    this.data = Arrays.copyOf(data, data.length);
  }
  public StringRow(int length) {
    data = new String[length];
    Arrays.fill(data, null);
  }
  /** Convert another row to this type by calling {@link Object#toString()}. */
  public StringRow(final Row data) {
    this.data = new String[data.getRowLength()];
    for(int i = 0; i < this.data.length; i++)
      this.data[i] = data.get(i).toString();
  }

  @Override public String toString() {return Arrays.toString(data);}
  @Override public boolean equals(Object other) {
    if(this == other) return true;
    if(other == null) return false;
    return other instanceof StringRow row && Arrays.equals(data, row.data);
  }
  @Override public int hashCode() {return Arrays.hashCode(data);}

  public String[] toArray() {return Arrays.copyOf(data, data.length);}
  public Stream<String> stream() {return Arrays.stream(data);}
  public void set(int index, String value) throws IndexOutOfBoundsException {data[index] = value;}
  public synchronized void setSynchronized(int index, String value) throws IndexOutOfBoundsException {set(index, value);}
  
  @Override public Class<?>[] getColumnClasses() {
    Class<?>[] classes = new Class<?>[getRowLength()];
    Arrays.fill(classes, String.class);
    return classes;
  }
  @Override public Class<String> getColumnClass(int index) {return String.class;}
  @Override public int getRowLength() {return data.length;}
  @Override public String get(int index) throws IndexOutOfBoundsException {return data[index];}
  @Override public Optional<BigDecimal> getAsNumber(int index) throws IndexOutOfBoundsException {
    try {
      return Optional.of(new BigDecimal(data[index].trim()));
    } catch (NumberFormatException | NullPointerException e) {
      return Optional.empty();
    }
  }
  @Override public String apply(int index) throws IndexOutOfBoundsException {return get(index);}
  @Override public StringRow project(final int ... indices) throws IndexOutOfBoundsException {
    String[] newRow = new String[indices.length];
    for(int i = 0; i < indices.length; i++)
      newRow[i] = get(indices[i]);
    return new StringRow(newRow);
  }
}
