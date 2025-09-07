import java.io.*;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

public class DecisionTreeClassifier<R extends Row, C> implements Classifier<R, C> {
  /** Actual decision tree. */
  protected final DecisionTree<R, C, C> dtree;

  protected DecisionTreeClassifier(AttrKind[] attrKinds, String[] columnNames, int depthLimit, int realAttributeSplits, int minSamplesToSplit,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction,
      Function<? super Stream<Pair<R, C>>, ? extends C> summarizer) {
    dtree = new DecisionTree<>(attrKinds, columnNames, depthLimit, realAttributeSplits, minSamplesToSplit, summarizer, impurityFunction);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, String[] columnNames, int depthLimit, int realAttributeSplits, int minSamplesToSplit,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    this(attrKinds, columnNames, depthLimit, realAttributeSplits, minSamplesToSplit, impurityFunction, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, String[] columnNames, int depthLimit, int realAttributeSplits,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    this(attrKinds, columnNames, depthLimit, realAttributeSplits, 1, impurityFunction, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, int depthLimit, int realAttributeSplits,
      ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction) {
    this(attrKinds, null, depthLimit, realAttributeSplits, 1, impurityFunction, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, int depthLimit, int realAttributeSplits) {
    dtree = new DecisionTree<>(attrKinds, depthLimit, realAttributeSplits, DecisionTreeClassifier::summarizeResult);
  }

  public DecisionTreeClassifier(AttrKind[] attrKinds, int realAttributeSplits) {
    this(attrKinds, attrKinds.length + 1, realAttributeSplits);
  }

  public final int getNumAttributes() {
    return dtree.getRowLength();
  }

  public final AttrKind getAttrKind(int colIndex) {
    return dtree.getAttrKind(colIndex);
  }

  public int depthLimit() {
    return dtree.depthLimit;
  }

  public ToDoubleFunction<? super Map<?, ? extends Number>> impurityFunction() {
    return dtree.impurityFunction;
  }

  public void walkTree(Appendable out) throws IOException {
    dtree.walkTree(out);
  }

  @Override
  public void fit(R input, C trueOutput) {
    dtree.addDataPoint(input, trueOutput);
  }

  @Override
  public void finishFitting() {
    dtree.buildTree();
  }

  @Override
  public Optional<C> predict(R input) {
    try {
      return Optional.of(dtree.decide(input));
    } catch (NoSuchElementException ex) {
      return Optional.empty();
    }
  }

  /** Result summary when we reach the end of a branch. */
  public static <RType extends Row, Cls> Cls summarizeResult(Stream<Pair<RType, Cls>> branchData) {
    return Utils.valueCounts(branchData.unordered().parallel().map(Pair::second))
        .entrySet()
        .stream()
        //.parallelStream() //No real benefit to parallelize here
        .unordered()
        .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
        .orElseThrow(IllegalArgumentException::new).getKey();
  }
}
