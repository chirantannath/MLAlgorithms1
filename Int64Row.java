import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Default implementation of {@link Row} featuring only {@code long}-type
 * attributes.
 * After creation we can change attributes but not their number.
 * 
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
    data = new long[dataRow.getRowLength()];
    for (int i = 0; i < data.length; i++) {
      try {
        data[i] = dataRow.getAsNumber(i).orElseThrow().longValue();
      } catch (NoSuchElementException e) {
        throw new IllegalArgumentException("Not a number column at position " + i + " for row " + dataRow, e);
      }
    }
  }

  @Override
  public String toString() {
    return Arrays.toString(data);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other)
      return true;
    if (other == null)
      return false;
    return other instanceof Int64Row row && Arrays.equals(data, row.data);
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(data);
  }

  public long[] toArray() {
    return Arrays.copyOf(data, data.length);
  }

  public LongStream stream() {
    return Arrays.stream(data);
  }

  public long getAsLong(int index) throws IndexOutOfBoundsException {
    return data[index];
  }

  public void setAsLong(int index, long value) throws IndexOutOfBoundsException {
    data[index] = value;
  }

  public synchronized void setSynchronized(int index, long value) throws IndexOutOfBoundsException {
    setAsLong(index, value);
  }

  @Override
  public Class<?>[] getColumnClasses() {
    Class<?>[] classes = new Class<?>[getRowLength()];
    Arrays.fill(classes, Long.TYPE);
    return classes;
  }

  @Override
  public Class<Long> getColumnClass(int index) {
    return Long.TYPE;
  }

  @Override
  public int getRowLength() {
    return data.length;
  }

  @Override
  public Long get(int index) throws IndexOutOfBoundsException {
    return getAsLong(index);
  }

  @Override
  public Optional<Long> getAsNumber(int index) throws IndexOutOfBoundsException {
    return Optional.of(getAsLong(index));
  }

  @Override
  public long applyAsLong(int index) {
    return getAsLong(index);
  }

  @Override
  public Int64Row project(final int... indices) throws IndexOutOfBoundsException {
    long[] newRow = new long[indices.length];
    for (int i = 0; i < indices.length; i++)
      newRow[i] = getAsLong(indices[i]);
    return new Int64Row(newRow);
  }

  public Int64Row doBinaryOperation(LongBinaryOperator op, Int64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    long[] newRow = new long[getRowLength()];
    for (int i = 0; i < newRow.length; i++)
      newRow[i] = op.applyAsLong(getAsLong(i), rightOperand.getAsLong(i));
    return new Int64Row(newRow);
  }

  public Int64Row doUnaryOperation(LongUnaryOperator op) {
    long[] newRow = new long[getRowLength()];
    for (int i = 0; i < newRow.length; i++)
      newRow[i] = op.applyAsLong(getAsLong(i));
    return new Int64Row(newRow);
  }

  public double distanceEuclidean(Int64Row rightOperand) {
    return Math.sqrt(distanceEuclideanSquared(rightOperand));
  }

  public long distanceEuclideanSquared(Int64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    long sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = getAsLong(i) - rightOperand.getAsLong(i);
      sum += (step * step);
    }
    return sum;
  }

  public long distanceManhattan(Int64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    long sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = Math.abs(getAsLong(i) - rightOperand.getAsLong(i));
      sum += step;
    }
    return sum;
  }

  public long distanceChebyshev(Int64Row rightOperand) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    long max, step;
    final int length = getRowLength();
    if (length == 0)
      throw new IllegalStateException();
    max = Math.abs(getAsLong(0) - rightOperand.getAsLong(0));
    for (int i = 1; i < length; i++) {
      step = Math.abs(getAsLong(i) - rightOperand.getAsLong(i));
      if (step > max)
        max = step;
    }
    return max;
  }

  public double distanceMinkowski(Int64Row rightOperand, final double p) {
    if (getRowLength() != rightOperand.getRowLength())
      throw new IllegalArgumentException();
    double sum = 0, step;
    final int length = getRowLength();
    for (int i = 0; i < length; i++) {
      step = Math.abs(getAsLong(i) - rightOperand.getAsLong(i));
      sum += Math.pow(step, p);
    }
    return Math.pow(sum, 1D / p);
  }

  public static Int64Row concatenate(Int64Row... rows) {
    final long[][] data = new long[rows.length][];
    for (int i = 0; i < rows.length; i++)
      data[i] = rows[i].data;
    final long[] concatenated = Utils.concat(data);
    return new Int64Row(concatenated);
  }
}
