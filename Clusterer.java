import java.util.function.*;
import java.util.stream.*;

/** 
 * Interface for clustering systems (grouping data). Groups are always identified
 * by index (0, 1, 2, ...) OR -1 if no clustering could be done.
 * @param <IR> input vector type
 * @author chirantannath
 */
public interface Clusterer<IR extends Row> extends Function<Supplier<Stream<IR>>, Stream<Pair<IR, ClusterLabel<IR>>>> {
  /**
   * Group incoming data represented by a stream of rows. We use a supplier
   * because the algorithm may need to process data repeatedly, 
   * which means that the input sequence returned must be the same each time
   * (some algorithms may tolerate unordering changes).
   * Although the return
   * type suggests this is a lazily evaluated function, this is not necessary and 
   * evaluation may happen early (that is, this function may take time by itself,
   * in which case the output groups, also represented by streams, may be sourced from
   * {@link java.util.Collection#stream()}, {@link java.util.stream.Stream.Builder#build()}, 
   * et cetera).
   * @return Stream of patterns with assigned clustering labels starting from 0.
   */
  @Override Stream<Pair<IR, ClusterLabel<IR>>> apply(Supplier<Stream<IR>> inputsSupplier);
}
