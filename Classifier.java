import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.*;
import java.util.stream.*;
/**
 * Interface for classifiers (predicting discrete output from input).
 * @param <IR> Input row type on which to classify
 * @param <OP> Output classificiation result type
 * @author chirantannath
 */
public interface Classifier<IR extends Row, OP> {
  /** 
   * Train (further) on a <i>single</i> input-output combination. 
   * This always changes state of this model so is expected to be called
   * in a sequential (thread-safe) manner.
   */
  void fit(IR input, OP trueOutput);
  /** @see #fit(IR, OP) */
  default void fit(Pair<? extends IR, ? extends OP> truePair) {fit(truePair.first(), truePair.second());}
  /** '
   * Train (further) on a set of input-output combinations. 
   */
  default void fit(Iterator<? extends IR> inputs, Iterator<? extends OP> trueOutputs) {
    while(inputs.hasNext()) {
      final var input = inputs.next();
      if(!trueOutputs.hasNext()) throw new IllegalArgumentException();
      final var output = trueOutputs.next();
      fit(input, output);
    }
  }
  /** @see #fit(Iterator<? extends IR>, Iterator<? extends OP>) */
  default void fit(Iterator<? extends Pair<? extends IR, ? extends OP>> truePairs) {
    while(truePairs.hasNext())
      fit(truePairs.next());
  }
  /** 
   * This is called for final post-processing work after all fitting; 
   * model should not change state after this method finishes. 
   * 
   * <p>The default implementation does nothing.</p>
   */
  default void finishFitting() {}

  /** Predict for an input. This NEEDS to be thread-safe (not change state of model). */
  OP predict(IR input);
  /** 
   * (Lazily) predict for a set of inputs.
   * Default implementation behaves as if
   * <pre>
   * return inputs.map(this::predict);
   * </pre>
   * @see #predict(IR)
   */
  default Stream<? extends OP> predict(Stream<? extends IR> inputs) {
    return inputs.map(this::predict);
  }
  
  /** 
   * Count correct classifications (accuracy), with progress reports. 
   * This processes sequentially.
   * Put {@code testedCountConsumer = null} if no progress report is desired.
   */
  default long countCorrect(Iterator<? extends IR> inputs, Iterator<? extends OP> trueOutputs, LongConsumer testedCountConsumer) {
    long count = 0, correctCount = 0;
    while(inputs.hasNext()) {
      final var input = inputs.next();
      if(!trueOutputs.hasNext()) throw new IllegalArgumentException();
      final var output = trueOutputs.next();
      final var predicted = predict(input);
      if(testedCountConsumer != null) testedCountConsumer.accept(++count);
      if(Objects.equals(output, predicted)) ++correctCount;
    }
    return correctCount;
  }
  /** @see #countCorrect(Iterator, Iterator, LongConsumer)} */
  default long countCorrect(Iterator<? extends Pair<? extends IR, ? extends OP>> truePairs, LongConsumer testedCountConsumer) {
    long count = 0, correctCount = 0;
    while(truePairs.hasNext()) {
      final var pair = truePairs.next();
      final var predicted = predict(pair.first());
      if(testedCountConsumer != null) testedCountConsumer.accept(++count);
      if(Objects.equals(pair.second(), predicted)) ++correctCount;
    }
    return correctCount;
  }
  /** 
   * Count correct classifications (accuracy), with progress reports. 
   * This processes in parallel.
   * Put {@code testedCountConsumer = null} if no progress report is desired.
   */
  default long countCorrectParallel(Stream<? extends Pair<? extends IR, ? extends OP>> truePairs, LongConsumer testedCountConsumer) {
    final var count = new AtomicLong(0);
    var stream = truePairs.unordered().parallel();
    if(testedCountConsumer != null) stream = stream.peek(pair -> testedCountConsumer.accept(count.incrementAndGet()));
    return stream.filter(pair -> Objects.equals(pair.second(), predict(pair.first()))).count();
  }
}
