import java.util.*;
import java.util.function.*;
import java.util.stream.*;
/**
 * Default implementation of {@link Row} featuring only {@code long}-type attributes.
 * After creation we can change attributes but not their number.
 * @author chirantannath
 */
public final class Int64Row implements Row, IntToLongFunction {
  private final long[] data;
  
  public Int64Row(long[] data) {
    Objects.requireNonNull(data);
    this.data = Arrays.copyOf(data, data.length);
  }
  public Int64Row(int length) {
    data = new long[length];
    Arrays.fill(data, 0l);
  }
  /** Tries to convert an arbitrary row into this class. */
  public Int64Row(final Row dataRow) throws IllegalArgumentException {
    data = IntStream.range(0, dataRow.getRowLength())
    .mapToLong(i -> dataRow.getAsNumber(i).orElseThrow(() -> new IllegalArgumentException("Not a number column at position "+i+" for row "+dataRow)).longValue())
    .toArray();
  }

  @Override public String toString() {return Arrays.toString(data);}
  @Override public boolean equals(Object other) {
    if(this == other) return true;
    if(other == null) return false;
    return other instanceof Int64Row row && Arrays.equals(data, row.data);
  }
  @Override public int hashCode() {return Arrays.hashCode(data);}

  public long[] toArray() {return Arrays.copyOf(data, data.length);}
  public LongStream stream() {return Arrays.stream(data);}
  public long getAsLong(int index) throws IndexOutOfBoundsException {return data[index];}
  public void setAsLong(int index, long value) throws IndexOutOfBoundsException {data[index] = value;}
  public synchronized void setSynchronized(int index, long value) throws IndexOutOfBoundsException {setAsLong(index, value);}

  @Override public Class<?>[] getColumnClasses() {
    return IntStream.range(0, data.length)
    .mapToObj(i -> (Class<?>)long.class).toArray(Class<?>[]::new);
  }
  @Override public Class<Long> getColumnClass(int index) {return Long.TYPE;}
  @Override public int getRowLength() {return data.length;}
  @Override public Long get(int index) throws IndexOutOfBoundsException {return getAsLong(index);}
  @Override public Optional<Long> getAsNumber(int index) throws IndexOutOfBoundsException {return Optional.of(getAsLong(index));}
  @Override public long applyAsLong(int index) {return getAsLong(index);}
  @Override public Int64Row project(final int ... indices) throws IndexOutOfBoundsException {
    return new Int64Row(Arrays.stream(indices).mapToLong(this::getAsLong).toArray());
  }

  public Int64Row doBinaryOperation(LongBinaryOperator op, Int64Row rightOperand) {
    if(getRowLength() != rightOperand.getRowLength()) throw new IllegalArgumentException();
    return new Int64Row(IntStream.range(0, getRowLength())
    .mapToLong(i -> op.applyAsLong(getAsLong(i), rightOperand.getAsLong(i)))
    .toArray());
  }

  public Int64Row doUnaryOperation(LongUnaryOperator op) {
    return new Int64Row(IntStream.range(0, getRowLength())
    .mapToLong(i -> op.applyAsLong(getAsLong(i)))
    .toArray());
  }
}
