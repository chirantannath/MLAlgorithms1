import java.util.function.*;

public record Float64Interval(double lowerBound, double higherBound) implements DoublePredicate {
  public Float64Interval(double lowerBound, double higherBound) {
    if (Double.compare(lowerBound, higherBound) > 0)
      throw new IllegalArgumentException();
    this.lowerBound = lowerBound;
    this.higherBound = higherBound;
  }

  /** Tests whether {@code value} falls into this bin or not. */
  @Override
  public boolean test(double value) {
    return Double.compare(lowerBound, value) <= 0 && Double.compare(value, higherBound) <= 0;
  }

  /** Tests whether {@code value} falls into this bin or not. */
  public boolean testForOptional(java.util.Optional<? extends Number> value) {
    return test(value.orElseThrow(() -> new IllegalArgumentException("expected number")).doubleValue());
  }

  /** Size (of the span) of this interval. */
  public double spanSize() {
    return higherBound - lowerBound;
  }

  /**
   * Splits this interval into {@code partitions.length + 1} (unequal) partitions,
   * each partition fraction must be between 0 and 1 and the sum must be less than
   * 1.
   */
  public Float64Interval[] split(double... partitions) {
    if (partitions.length == 0)
      return new Float64Interval[] { this };
    final double totalSize = spanSize();

    final int numPartitions = partitions.length + 1;
    final var partitionBounds = new Float64Interval[numPartitions];

    double currentAllocated = 0;
    for (int i = 0; i < partitions.length; i++) {
      if (Double.compare(partitions[i], 0) < 0 || Double.compare(partitions[i], 1) > 0)
        throw new IllegalArgumentException("partition number " + i);
      final double partitionSize = totalSize * partitions[i];
      final double partitionStart = lowerBound + currentAllocated;
      final double partitionEnd = partitionStart + partitionSize;
      if (Double.compare(partitionSize, 0) < 0 || Double.compare(partitionEnd, higherBound) > 0)
        throw new IllegalArgumentException("partition number " + i);
      partitionBounds[i] = new Float64Interval(partitionStart, partitionEnd);
      currentAllocated += partitionSize;
    }
    partitionBounds[numPartitions - 1] = new Float64Interval(lowerBound + currentAllocated, higherBound);

    return partitionBounds;
  }

  public Float64Interval[] split(int parts) {
    if (parts < 0)
      throw new IllegalArgumentException();
    else if (parts == 0)
      return new Float64Interval[] {};
    else if (parts == 1)
      return new Float64Interval[] { this };

    final double totalSize = spanSize();
    final double partitionSize;
    {
      final int log2 = Utils.perfectLog2(parts);
      partitionSize = log2 < 0 ? totalSize / parts : Math.scalb(totalSize, -log2);
    }
    final var partitionBounds = new Float64Interval[parts];
    partitionBounds[0] = new Float64Interval(lowerBound, lowerBound + partitionSize);
    for (int i = 1; i < (parts - 1); i++) {
      final double start = partitionBounds[i - 1].higherBound;
      final double end = start + partitionSize;
      partitionBounds[i] = new Float64Interval(start, end); 
    }
    partitionBounds[parts - 1] = new Float64Interval(partitionBounds[parts - 2].higherBound, higherBound);

    return partitionBounds;
  }
}