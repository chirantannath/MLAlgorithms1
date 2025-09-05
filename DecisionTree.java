import java.util.*;
import java.util.function.*;
import java.util.stream.*;

/**
 * Decision Tree main building class. Data points are {@link Pair}s of
 * {@code R}s and {@code IntermediateType}s, and the decision tree must arrive
 * at a result of type ResultType.
 * 
 * @author chirantannath
 */
public class DecisionTree<R extends Row, IntermediateType, ResultType> {
  protected abstract sealed class Node permits AttrNode, ResultNode {
    /** Parent from which this node came from. */
    final Node parent;
    /**
     * Filtering condition on which this branch was taken; constant predicate for
     * root nodes.
     */
    final Predicate<R> branchFilter;
    /** Depth of this node in tree, 0 for root. */
    final int depth;
    /** Filled in during training. */
    Collection<? extends Node> children = null;

    Node(Node parent, Predicate<R> branchFilter) {
      this.parent = parent;
      this.branchFilter = branchFilter == null ? Utils.constantPredicate(true) : branchFilter;
      depth = parent == null ? 0 : parent.depth + 1;
    }

    Node(Node parent) {
      this(parent, null);
    }

    Node() {
      this(null);
    }

    final boolean isRoot() {
      return parent == null;
    }

    abstract boolean isChild();

    /** This function is for use during training. */
    final boolean filterFromRoot(R row) {
      for (Node node = this; node != null; node = node.parent)
        if (!node.branchFilter.test(row))
          return false;
      return true;
    }

    /**
     * This function is for use during prediction. Must NOT fail in any case (must
     * return at least one branch), except if this node is a child, in which case
     * this function MUST RETURN {@code null}.
     */
    abstract Predicate<R> getChildBranch(R row);

    /**
     * This function is for use during building the tree. Return all possible child
     * branches (processing may happen here, and the result may be cached by
     * implementations, this is not subject to change). Must always return some
     * array (even if is is of length 0).
     */
    abstract Predicate<R>[] getAllChildBranches();
  }

  /** Nodes that split on the value of a single attribute. */
  protected abstract sealed class AttrNode extends Node permits CategoricalAttrNode, RealAttrNode {
    final int attrIndex;

    AttrNode(int attrIndex, Node parent, Predicate<R> branchFilter) {
      super(parent, branchFilter);
      Objects.checkIndex(attrIndex, getRowLength());
      this.attrIndex = attrIndex;
    }

    AttrNode(int attrIndex, Node parent) {
      this(attrIndex, parent, null);
    }

    AttrNode(int attrIndex) {
      this(attrIndex, null);
    }
  }

  /** Nodes splitting on categorical (discrete) value attributes. */
  protected final class CategoricalAttrNode extends AttrNode {
    final Map<?, Predicate<R>> categories;

    CategoricalAttrNode(int attrIndex, Node parent, Predicate<R> branchFilter, Set<?> attrValues) {
      super(attrIndex, parent, branchFilter);
      Objects.requireNonNull(attrValues, "attrValues");
      final Map<Object, Predicate<R>> branches = new HashMap<>();
      for (final var value : attrValues)
        branches.put(value, row -> Objects.equals(row.get(attrIndex), value));
      categories = Collections.unmodifiableMap(branches);
    }

    CategoricalAttrNode(int attrIndex, Node parent, Predicate<R> branchFilter) {
      this(attrIndex, parent, branchFilter,
          categoricalValueCounts(attrIndex).keySet());
    }

    CategoricalAttrNode(int attrIndex, Node parent) {
      this(attrIndex, parent, null);
    }

    CategoricalAttrNode(int attrIndex) {
      this(attrIndex, null);
    }

    @Override
    boolean isChild() {
      return false;
    }

    @Override
    Predicate<R> getChildBranch(R row) {
      return categories.get(row.get(attrIndex));
    }

    @Override
    @SuppressWarnings("unchecked")
    Predicate<R>[] getAllChildBranches() {
      return categories.values().toArray((Predicate<R>[]) new Predicate<?>[categories.size()]);
    }
  }

  /**
   * Nodes splitting on continuous (real) value attributes. UNIMPLEMENTED AS OF
   * NOW.
   */
  protected final class RealAttrNode extends AttrNode {

    RealAttrNode(int attrIndex, Node parent, Predicate<R> branchFilter) {
      super(attrIndex, parent, branchFilter);
    }

    RealAttrNode(int attrIndex, Node parent) {
      super(attrIndex, parent);
    }

    RealAttrNode(int attrIndex) {
      super(attrIndex);
    }

    @Override
    boolean isChild() {
      return false;
    }

    @Override
    Predicate<R> getChildBranch(R row) {
      // TODO Auto-generated method stub
      throw new UnsupportedOperationException("Unimplemented method 'getChildBranch'");
    }

    @Override
    Predicate<R>[] getAllChildBranches() {
      throw new UnsupportedOperationException();
    }
  }

  protected final class ResultNode extends Node {
    /** The result stored. May be {@code null} or any other arbitrary object. */
    final ResultType result;

    ResultNode(ResultType result, Node parent, Predicate<R> branchFilter) {
      super(parent, branchFilter);
      this.result = result;
    }

    ResultNode(ResultType result, Node parent) {
      this(result, parent, null);
    }

    ResultNode(ResultType result) {
      this(result, null);
    }

    ResultNode() {
      this(null, null, null);
    }

    /** Creates an empty "stop" node. */
    ResultNode(Node parent) {
      this(null, parent, null);
    }

    @Override
    boolean isChild() {
      return true;
    }

    @Override
    Predicate<R> getChildBranch(R row) {
      return null;
    }

    @Override
    @SuppressWarnings("unchecked")
    Predicate<R>[] getAllChildBranches() {
      return (Predicate<R>[]) EMPTY_PREDICATES;
    }

    public static Predicate<?>[] EMPTY_PREDICATES = new Predicate<?>[0];
  }

  protected final List<Pair<R, IntermediateType>> rootData = new ArrayList<>();
  protected final AttrKind[] attrKinds;
  public final int depthLimit;
  public final ToDoubleFunction<? super Stream<?>> impurityFunction;
  public final Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer;

  /** The decision tree root. */
  protected Node treeRoot = null; // Will be assigned later

  public DecisionTree(AttrKind[] attrKinds, int depthLimit,
      Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer,
      ToDoubleFunction<? super Stream<?>> impurityFunction) {
    Objects.requireNonNull(attrKinds, "attrKinds");
    Objects.requireNonNull(impurityFunction, "impurityFunction");
    Objects.requireNonNull(summarizer, "summarizer");
    this.attrKinds = Arrays.copyOf(attrKinds, attrKinds.length);
    if (depthLimit <= 0)
      throw new IllegalArgumentException("depthLimit");
    this.depthLimit = depthLimit;
    this.impurityFunction = impurityFunction;
    this.summarizer = summarizer;
  }

  public DecisionTree(AttrKind[] attrKinds, int depthLimit,
      Function<? super Stream<Pair<R, IntermediateType>>, ? extends ResultType> summarizer) {
    this(attrKinds, depthLimit, summarizer, Utils::countedEntropy);
  }

  public void addDataPoint(R input, IntermediateType intermediate) {
    rootData.add(new Pair<>(input, intermediate));
  }

  /** Get a part of the root dataset. */
  protected final Stream<Pair<R, IntermediateType>> filteredData(final Predicate<R> filter) {
    return rootData
      //.stream()
      .parallelStream() //Currently disabled for debugging
      .filter(p -> filter.test(p.first()));
  }

  /** FOR CATEGORICAL VALUES ONLY, get all value counts from full root dataset. */
  protected final Map<?, Long> categoricalValueCounts(int attrIndex) {
    Map<?, Long> result;
    if ((result = categoricalValueCountsCache.get(attrIndex)) != null)
      return result;
    result = Utils.valueCounts(rootData.parallelStream().unordered().map(p -> p.first().get(attrIndex)));
    categoricalValueCountsCache.put(attrIndex, result);
    return result;
  }

  protected final Map<Integer, Map<?, Long>> categoricalValueCountsCache = new HashMap<>();

  /** On which attribute should I split this set? */
  protected int findSplittingAttribute(final Predicate<R> setMembership, Iterator<Integer> attributeIndexes) {
    final var totalLength = filteredData(setMembership).unordered().count();
    if (totalLength == 0)
      return -1;
    final double rootImpurity = impurityFunction
        .applyAsDouble(
            filteredData(setMembership).unordered().map(Pair::second));

    if (attributeIndexes == null)
      attributeIndexes = IntStream.range(0, getRowLength()).boxed().iterator();

    double currentGain = Double.NEGATIVE_INFINITY;
    int maxGainIndex = -1;
    while (attributeIndexes.hasNext()) {
      final int attrIndex = attributeIndexes.next();
      double attrImpurity = 0;
      switch (attrKinds[attrIndex]) {
        case CONTINUOUS -> throw new UnsupportedOperationException(
            "continuous values unsupported as of now (at index " + attrIndex + ")");
        case CATEGORICAL -> {
          // Split on attribute
          final var valueCounts = Utils
              .valueCounts(filteredData(setMembership).map(p -> p.first().get(attrIndex)).unordered());
          for (final var vcount : valueCounts.entrySet())
            attrImpurity += impurityFunction
                .applyAsDouble(filteredData(setMembership.and(r -> Objects.equals(r.get(attrIndex), vcount.getKey())))
                    .unordered().map(Pair::second))
                * vcount.getValue() / totalLength;
          // break; Not required in rule switch
        }
      }
      final double gain = rootImpurity - attrImpurity; // impurity should decrease
      if (gain > currentGain) {
        currentGain = gain;
        maxGainIndex = attrIndex;
      }
    }

    return maxGainIndex;
  }

  /**
   * Build tree given root, current level number and attribute set already
   * selected. Note that {@code attributesSelected} needs to be a modifiable set.
   */
  protected void buildTree(final Node root, final int currentLevel, final Set<Integer> attributesSelected) {
    if (root == null || root.isChild())
      return;
    final Collection<Integer> attributesToBranch = IntStream.range(0, getRowLength()).boxed()
        .filter(i -> !attributesSelected.contains(i))
        .collect(Collectors.toUnmodifiableSet());
    final var children = new ArrayList<Node>();
    for (final var branch : root.getAllChildBranches()) {
      final var branchDataFilter = branch.and(root::filterFromRoot);
      // Check if we have at least one data point in this branch
      if (filteredData(branchDataFilter).unordered().findAny().isEmpty())
        continue;
      final Node node;
      if (currentLevel <= depthLimit && !attributesToBranch.isEmpty()) { // I can split further from here.
        final int attrIndex = findSplittingAttribute(branchDataFilter, attributesToBranch.iterator());
        if (attrIndex < 0)
          continue;
        node = switch (attrKinds[attrIndex]) {
          case CONTINUOUS -> throw new UnsupportedOperationException();
          case CATEGORICAL -> new CategoricalAttrNode(attrIndex, root, branch);
        };
      } else { // Result node calculation, either on reaching depth or when no more attributes
               // to branch on
        final var result = summarizer.apply(filteredData(branchDataFilter));
        node = new ResultNode(result, root, branch);
      }
      children.add(node);
    }
    assert !children.isEmpty();
    root.children = Collections.unmodifiableCollection(children);

    // Recursively build tree further.
    for (final Node child : children) {
      if (child instanceof AttrNode attrNode) {
        attributesSelected.add(attrNode.attrIndex);
        buildTree(attrNode, currentLevel + 1, attributesSelected);
        attributesSelected.remove(attrNode.attrIndex);
      }
    }
  }

  public void buildTree() {
    // Step 1. Find a splitting attribute.
    final int rootAttrIndex = findSplittingAttribute(Utils.constantPredicate(true), null);
    assert rootAttrIndex >= 0;
    treeRoot = switch (attrKinds[rootAttrIndex]) {
      case CONTINUOUS -> throw new UnsupportedOperationException();
      case CATEGORICAL -> new CategoricalAttrNode(rootAttrIndex);
    };
    final var attributesSelected = new Utils.WholeNumbersSet();
    attributesSelected.add(rootAttrIndex);
    buildTree(treeRoot, 1, attributesSelected);
  }

  public ResultType predict(final R input) {
    if (treeRoot == null)
      throw new IllegalStateException();
    Node current = treeRoot;
    while (true) {
      if (current instanceof ResultNode r)
        return r.result;
      assert current != null && !current.isChild();
      Node newCurrent = current.children
          .stream() //No need to parallelize here.
          //.parallelStream()
          .unordered()
          .filter(child -> child.branchFilter.test(input))
          .findAny().orElseThrow(() -> new NoSuchElementException("Unable to predict for "+input));
      assert !Objects.equals(current, newCurrent);
      current = newCurrent;
    }
  }

  public int getRowLength() {
    return attrKinds.length;
  }

  public AttrKind getAttrKind(int colIndex) {
    return attrKinds[colIndex];
  }
}
