package TransferLearning;

import GridWorld.*;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.behavior.functionapproximation.sparse.LinearVFA;
import burlap.behavior.functionapproximation.sparse.SparseStateFeatures;
import burlap.behavior.functionapproximation.sparse.StateFeature;
import burlap.behavior.functionapproximation.sparse.tilecoding.Tiling;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.debugtools.RandomFactory;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.util.*;


/**
 * A feature database using CMACs [1] AKA Tiling Coding for states that are first converted into a feature vector. Because States are converted into a feature vector
 * before tiling them.
 * <p>
 * Different tilings can be created over different dimensions of the converted state feature vector and different tiling widths for each dimension can be specified. Each tiling
 * over the same dimensions can either be randomly jittered from each other or uniformly distributed across the space, which is specified using the {@link TilingArrangement}
 * enumerator.
 * <p>
 * To specify the tiling used, use the {@link #addTilingsForAllDimensionsWithWidths(double[], int, TilingArrangement)} or
 * {@link #addTilingsForDimensionsAndWidths(boolean[], double[], int, TilingArrangement)} method.
 * 
 * 
 * <p>
 * 
 * 1. Albus, James S. "A theory of cerebellar function." Mathematical Biosciences 10.1 (1971): 25-61
 * 
 * @author James MacGlashan
 *
 */
public class TLTileCodingFeatures implements SparseStateFeatures {

	/**
	 * The generator that turns OO-MDP state objects into state feature vectors.
	 */
	protected DenseStateFeatures 												featureVectorGenerator;
	
	
	
	/**
	 * A random object for jittering the tile alignments.
	 */
	protected Random															rand = RandomFactory.getMapped(0);
	
	/**
	 * A list of all the tilings used.
	 */
	List<Tiling>																tilings;
	
	
	/**
	 * Mapping to state features
	 */
	List<Map<Tiling.FVTile, Integer>>	 										stateFeatures;

	
	/**
	 * The identifier to use for the next state feature.
	 */
	protected int																nextStateFeatureId = 0;

	
	/**
	 * 
	 */
	HashMap<boolean [], TilingsLayout>											masksNwidths = new HashMap<boolean [], TilingsLayout>();
	
	/**
	 * This map is for mapping the tile IDs to the tilings
	 */
	public HashMap<Integer,ArrayList<Integer>>									tilingsNtiles = new HashMap<Integer, ArrayList<Integer> >();		
	
	/**
	 * We suppose all the tilings created specifically for a new task are added at the beginning of the process, therefore we keep track of this index
	 * for calculating the covering given by the transfer process in a second moment
	 */
	private int																	notTransferTilings = 0;
	
	private int																	transferCovered = 0;
	private int																	transferNotCovered = 0;
	
	/** this variable makes sure that, when false, the creation of new State or State-Action features is stopped 
	 * so that we can make tests on different states without making it seem as a learning/exploration "visit" */
	private boolean 															learning = true;
	

	
	@Override
	public TLTileCodingFeatures copy() {
		TLTileCodingFeatures tilecoding = new TLTileCodingFeatures(this.featureVectorGenerator);
		tilecoding.rand = this.rand;
		tilecoding.tilings = new ArrayList<Tiling>(this.tilings);

		tilecoding.stateFeatures = new ArrayList<Map<Tiling.FVTile, Integer>>(this.stateFeatures.size());
		for(Map<Tiling.FVTile, Integer> el : this.stateFeatures){
			Map<Tiling.FVTile, Integer> nel = new HashMap<Tiling.FVTile, Integer>(el);
			tilecoding.stateFeatures.add(nel);
		}

		tilecoding.nextStateFeatureId = this.nextStateFeatureId;

		return tilecoding;
	}

	/**
	 * Initializes specifying the kind of state feature vector generator to use for turning OO-MDP states into feature vectors.
	 * The resulting feature vectors are what is tiled by this class.
	 * @param featureVectorGenerator the OO-MDP state to feature vector generator to use
	 */
	public TLTileCodingFeatures(DenseStateFeatures featureVectorGenerator){
		
		this.featureVectorGenerator = featureVectorGenerator;
		this.tilings = new ArrayList<Tiling>();
		this.stateFeatures = new ArrayList<Map<Tiling.FVTile,Integer>>();
		
	}
	
	
	/**
	 * Adss a number of tilings where each tile is dependent on the dimensions that are labeled as "true" in the dimensionMask parameter. The widths parameter
	 * specifies the width of each tile along that given dimension. If tileArrangement is set to {@link TilingArrangement#UNIFORM} then each of the nTilings
	 * created with will be uniformly spaced across the width of each dimension. If it is set to {@link TilingArrangement#RANDOM_JITTER} then each tiling
	 * will be offset by a random amount.
	 * @param dimensionMask each true entry in this boolen array is a dimension over which the tiling will be defined.
	 * @param widths the width of tiles along each dimension. This value should be non-zero for each dimension unless the tiling doesn't depend on that dimension.
	 * @param nTilings the number of tilings over the specified dimensions to create
	 * @param tileArrangement whether the created tiles are uniformally spaced or randomly spaced.
	 */
	public void addTilingsForDimensionsAndWidths(boolean [] dimensionMask, double [] widths, int nTilings, TilingArrangement tileArrangement){
		
		int initialSize = tilings.size();
		
		if(initialSize == 0)
			this.notTransferTilings = nTilings - 1;//we save the index not the number of tilings
		
		for(int i = 0; i < nTilings; i++){
			this.stateFeatures.add(new HashMap<Tiling.FVTile, Integer>());
			double [] offset;
			if(tileArrangement == TilingArrangement.RANDOM_JITTER){
				offset = this.produceRandomOffset(dimensionMask, widths);
			}
			else{
				offset = this.produceUniformTilingsOffset(dimensionMask, widths, i, nTilings);
			}
			Tiling tiling = new Tiling(widths, offset, dimensionMask);
			this.tilings.add(tiling);
			this.tilingsNtiles.put(i+initialSize, new ArrayList<Integer>());
		}
		
		TilingsLayout currentTilingLayout = new TilingsLayout(widths, nTilings, initialSize);
		this.masksNwidths.put(dimensionMask, currentTilingLayout);
		
	}
	
	
	/**
	 * Adss a number of tilings where each tile is dependent on *all* the dimensions of a state feature vector. The widths parameter
	 * specifies the width of each tile along that given dimension. If tileArrangement is set to {@link TilingArrangement#UNIFORM} then each of the nTilings
	 * created with will be uniformly spaced across the width of each dimension. If it is set to {@link TilingArrangement#RANDOM_JITTER} then each tiling
	 * will be offset by a random amount.
	 * @param widths the width of tiles along each dimension. This value should be non-zero for each dimension .
	 * @param nTilings the number of tilings over the specified dimensions to create.
	 * @param tileArrangement whether the created tiles are uniformally spaced or randomly spaced.
	 */
	public void addTilingsForAllDimensionsWithWidths(double [] widths, int nTilings, TilingArrangement tileArrangement){
		
		boolean [] dimensionMask = new boolean[widths.length];
		for(int i = 0; i < dimensionMask.length; i++){
			dimensionMask[i] = true;
		}
		this.addTilingsForDimensionsAndWidths(dimensionMask, widths, nTilings, tileArrangement);
		
	}
	
	@Override
	public List<StateFeature> features(State s) {
		
		double [] input = this.featureVectorGenerator.features(s);
		List<StateFeature> features = new ArrayList<StateFeature>();
		
		//start the counting from zero every time
		this.transferNotCovered = 0;
		this.transferCovered = 0;
		
		for(int i = 0; i < this.tilings.size(); i++){
			Tiling tiling = this.tilings.get(i);
			Map<Tiling.FVTile, Integer> tileFeatureMap = this.stateFeatures.get(i);
			
			Tiling.FVTile tile = tiling.getFVTile(input);
				
			int f = this.getOrGenerateFeature(tileFeatureMap, tile, i, tiling);
			
			if(this.learning){
				if(!this.tilingsNtiles.get(i).contains(f))//if the tiling I does not contain the tile F yet...
					this.tilingsNtiles.get(i).add(f);
				
				StateFeature sf = new StateFeature(f, 1.);
				features.add(sf);
			}else{
				StateFeature sf;
				
				if(f == -1){
					sf = new StateFeature(-1, 0);//dummy state feature || NOT TO BE PROCESSED
				}else{
					sf = new StateFeature(f, 1.);
				}
				features.add(sf);
			}
		}

		return features;
	}

//	public void setFeatureVectorGenerator(DenseStateFeatures newFVG){
//		this.featureVectorGenerator = newFVG; 
//	}
	
	@Override
	public int numFeatures() {
		return nextStateFeatureId;
	}
	
	public int numTIlings(){
		return this.tilings.size();
	}

	/**
	 * Returns the stored feature id or creates, stores and returns one. If a feature id is created, then the {@link #nextStateFeatureId} data member of this
	 * object is incremented.
	 * @param tileFeatureMap the map from tiles to feature ids
	 * @param tile the tile for which a feature id is returned.
	 * @return the feature id for the tile.
	 */
	protected int getOrGenerateFeature(Map<Tiling.FVTile, Integer> tileFeatureMap, Tiling.FVTile provTile, int currTilingNum, Tiling currTiling){
		Tiling.FVTile tile = null;
		
		int [] tiledVector = provTile.tiledVector;
		int [] suppTiledVector = new int [tiledVector.length];
		boolean [] suppMask = new boolean [tiledVector.length];
		double [] suppWidths = new double [tiledVector.length];
		int [] newTiledVector;
		boolean [] newMask;
		double [] newWidths, newOff;
		
		for(boolean [] mask : this.masksNwidths.keySet()){
			TilingsLayout layout = this.masksNwidths.get(mask);
			
			int firstTiling = layout.getFirstTiling();
			int nTilings = layout.getNumTilings();
			
			//Matching tilings, AKA does @param mask span the range of tilings to which @param currTiling belongs?
			if(currTilingNum >= firstTiling && currTilingNum < firstTiling + nTilings){
				double [] widths = layout.getWidths();
				
				int j = 0;
				//Fill the support vector which will have j useful values (the first j ones)
				//and i-j meaningless 0s (the remaining elements in the vector)
				for(int i = 0; i < mask.length; ++i){
					if(mask[i] == false && widths[i] == 0)
						continue;
					else{
						suppMask[j] = mask[i];
						suppWidths[j] = widths[i];
						suppTiledVector[j] = tiledVector[i];
						++j;
					}
				}	
				
				//If j and i change asincronously we need to cut off the last i-j elements from the vector 
				if(j != mask.length){
					newTiledVector = new int [j];
					newMask = new boolean[j];
					newWidths = new double[j];
					newOff = new double[j];
					
					for(int k = 0; k<j; ++k) {
						newTiledVector[k] = suppTiledVector[k]; 
						newMask[k] = suppMask[k]; 
						newWidths[k] = suppWidths[k]; 
						newOff[k] = 0.; 
					}	
					
					tile = new Tiling(newWidths, newOff, newMask).new FVTile(newTiledVector);
//					tile = currTiling.new FVTile(newTiledVector);
					
				}else
					tile = provTile;
			}	
		}
		
		Integer stored = tileFeatureMap.get(tile);
		if(stored == null){
			this.transferNotCovered += 1;
			
			if(this.learning){//these actions are proper of the learning time, therefore if the learning variable is "off" they must not be performed
				stored = this.nextStateFeatureId;
				tileFeatureMap.put(tile, stored);
				this.nextStateFeatureId++;
			}else
				stored = -1;
		}else{
			this.transferCovered += 1;
		}

		return stored;
	}
	
	public void setLearning(boolean tf){
		this.learning = tf;
	}
	
	/**
	 * Returns or creates stores and returns the list of action feature ids in the given map for the given tile. If a new list is created, it will
	 * be empty and will need to action features added to it.
	 * @param tileFeatureMap the map from tiles for a state to the list of action features for that state 
	 * @param tile the tile for which the list of action features is returned.
	 * @return the list of action features.
	 */
	protected List<ActionFeatureID> getOrGenerateActionFeatureList(Map<Tiling.FVTile, List<ActionFeatureID>> tileFeatureMap, Tiling.FVTile tile){
		
		List<ActionFeatureID> stored = tileFeatureMap.get(tile);
		if(stored == null){
			stored = new ArrayList<TLTileCodingFeatures.ActionFeatureID>();
			tileFeatureMap.put(tile, stored);
		}
		return stored;
		
	}


	
	
	/**
	 * After all the tiling specifications have been set, this method can be called to produce a linear
	 * VFA object.
	 * @param defaultWeightValue the default value weights for the CMAC features will use.
	 * @return a linear ValueFunctionApproximation object that uses this feature database
	 */
	public LinearVFA generateVFA(double defaultWeightValue){
		return new LinearVFA(this, defaultWeightValue);
	}

	
	
	/**
	 * Creates and returns a random tiling offset for the given widths and required dimensions.
	 * @param dimensionMask each true entry is a dimension on which the tiling depends
	 * @param widths the width of each dimension
	 * @return a random tiling offset.
	 */
	protected double [] produceRandomOffset(boolean [] dimensionMask, double [] widths){
		double [] offset = new double[dimensionMask.length];
		
		for(int i = 0; i < offset.length; i++){
			if(dimensionMask[i]){
				offset[i] = this.rand.nextDouble()*widths[i];
			}
			else{
				offset[i] = 0.;
			}
		}
		
		return offset;
	}
	
	
	/**
	 * Creates and returns an offset that is uniformly spaced from other tilings.
	 * @param dimensionMask each true entry is a dimension on which the tiling depends
	 * @param widths widths the width of each dimension
	 * @param ithTiling which tiling of the nTilings for which this offset is to be generated
	 * @param nTilings the total number of tilings that will be uniformaly spaced
	 * @return an offset that is uniformly spaced from other tilings.
	 */
	protected double [] produceUniformTilingsOffset(boolean [] dimensionMask, double [] widths, int ithTiling, int nTilings){
		double [] offset = new double[dimensionMask.length];
		
		for(int i = 0; i < offset.length; i++){
			if(dimensionMask[i]){
				offset[i] = ((double)ithTiling / (double)nTilings)*widths[i];
			}
			else{
				offset[i] = 0.;
			}
		}
		
		return offset;
	}
	
	
	/**
	 * Returns the {@link ActionFeatureID} with an equivalent {@link Action} in the given list or null if there is none.
	 * @param actionFeatures the list of {@link ActionFeatureID} objects to search.
	 * @param forAction the {@link Action} for which a match is to be found.
	 * @return the {@link ActionFeatureID} with an equivalent {@link Action} in the given list or null if there is none.
	 */
	protected ActionFeatureID matchingActionFeature(List<ActionFeatureID> actionFeatures, Action forAction){
		
		for(ActionFeatureID aid : actionFeatures){
			if(aid.ga.equals(forAction)){
				return aid;
			}
		}
		
		return null;
	}
	
	
	/**
	 * A class for associating a {@link Action} with a feature id.
	 * @author James MacGlashan
	 *
	 */
	protected class ActionFeatureID{
		public int id;
		public Action ga;
		
		public ActionFeatureID(Action ga, int id){
			this.id = id;
			this.ga = ga;
		}
		
	}
	
	
	/**
	 * Transfer Learning for State Features
	 * @param newTLTCFeatures are the new features on which we want to transfer the current knowledge
	 * @param maskForSelection is the mask for selecting the tilings we want to transfer. If null the transfer takes place for all the tilings
	 *
	 */
	public ArrayList<Integer> transferKnowledge(TLTileCodingFeatures newTLTCFeatures, boolean [] maskForSelection){
		
//		if(newTLTCFeatures.numFeatures() != 0){
//			System.out.println("The transfer of knowledge can happen iif the input features are still NOT learnt");
//			return null;
//		}	
		
		/** we start counting the tilings from the last one created for this new set */
		int lastNumCurrTlngs = newTLTCFeatures.tilings.size();
//		int oldLastNumCurrTlngs = lastNumCurrTlngs;
		
		ArrayList<Integer> indexToChange = new ArrayList<Integer>();
		
		/** In this for loop we skim "masksNwidths" key by key, which are masks, so to be able to transfer knowledge 
		 * Each Mask corresponds to a different set of tilings, like in the case of multiple sources. For the moment 
		 * transferring offsets is not necessary since the elements of each set of tilings are added all at a time, 
		 * but each set is like a different batch*/
		for(boolean [] mask : this.masksNwidths.keySet()){
			if(Arrays.equals(mask, maskForSelection) || maskForSelection == null){			
				Object[] matched = MatchValues(newTLTCFeatures.getKeys(), this.getKeys(), mask, this.masksNwidths.get(mask).getWidths());
				
				/** TilingArrangement is fixed to UNIFORM, we might need to find a more flexible way to set this parameter 
				 *  NOTE: Using matched[] as inputs for addTilingsForDimAndWid is the key of the transfer at this point
				*/
				int numCurrTlngs = this.masksNwidths.get(mask).getNumTilings(); 
				newTLTCFeatures.addTilingsForDimensionsAndWidths((boolean[])matched[0], (double[])matched[1], numCurrTlngs, TilingArrangement.UNIFORM);
				
				int tilingFrom = lastNumCurrTlngs;
				lastNumCurrTlngs += numCurrTlngs;
										
				for(int j = tilingFrom, k = this.masksNwidths.get(mask).getFirstTiling(); j <= lastNumCurrTlngs - 1; ++j, ++k){
					newTLTCFeatures.setStateFeatures(this.stateFeatures.get(k), j);
					indexToChange.addAll(this.tilingsNtiles.get(k));
				}	
			}	
			
			//just updating indexis
//			if(lastNumCurrTlngs == oldLastNumCurrTlngs){
//				lastNumCurrTlngs += this.masksNwidths.get(mask).getNumTilings();
//			}
//			oldLastNumCurrTlngs = lastNumCurrTlngs;
		}
		
		newTLTCFeatures.setNextStateFeatureId(this.nextStateFeatureId);
		
		return indexToChange;
		
	}
	
	/** This function takes the mask and the relative values for changing them of the current object
	 *  and sorts and extends them so to be appliable on the target vfa
	 * @param keys0 are the keys not to be changed 
	 * @param keys1 are the keys to be re-sorted 
	 * @param bools1 is the source mask
	 * @param vals1 is the source list of values
	 */
	private Object[] MatchValues(List<String> keys0, List<String> keys1, boolean [] bools1, double [] vals1){
		boolean [] b1 = new boolean [keys0.size()];
		double[] v1 = new double [keys0.size()];
		int index1;
		List<String> k1 = new ArrayList<String>(keys1);
		
		for(int index0 = 0; index0 < keys0.size(); ++index0){
			index1 = k1.indexOf(keys0.get(index0));
			
			if(index1 != -1){
				b1[index0] = bools1[index1];
				v1[index0] = vals1[index1];
			}else{
				b1[index0] = false;
				v1[index0] = 0;
			}	
		}
		
		return new Object[]{b1, v1};
	}
	
	public List<String> getKeys(){
		List<String> sortedKeys = null;
		
		if(this.featureVectorGenerator instanceof TLNormalizedVarFeatBlockDude)
			sortedKeys = ((TLNormalizedVarFeatBlockDude)this.featureVectorGenerator).getSortedKeys();
		else if(this.featureVectorGenerator instanceof TLNormalizedVarFeatGridWorld) {
			sortedKeys = ((TLNormalizedVarFeatGridWorld)this.featureVectorGenerator).getSortedKeys();
		}else if(this.featureVectorGenerator instanceof TLDefineRangeNormalizedVarFeatGridWorld) {
			sortedKeys = ((TLDefineRangeNormalizedVarFeatGridWorld)this.featureVectorGenerator).getSortedKeys();
		}else
			System.out.println("ERROR: Impossible to retrieve keys. featureVectorGenerator is not a TLNormalizedVarFeatBlockDude");
		
		return sortedKeys;
	}
	
	/** @param i is the tiling number which identifies the state features we are setting */
	public void setStateFeatures(Map<Tiling.FVTile, Integer> sourceStateFeatures, int i){
		this.stateFeatures.set(i, new HashMap<Tiling.FVTile, Integer> (sourceStateFeatures));
	}
	
	public void setNextStateFeatureId(int nsfi){
		this.nextStateFeatureId = nsfi;
	}
	
	public int getTransferCovered(){
		return this.transferCovered;
	}
	
	public int getTransferNotCovered(){
		return this.transferNotCovered;
	}
	
	/** Class for coupling the number of layers with the relative widths */
	public class TilingsLayout{
		
		private double [] widths;
		private int numTilings;
		private int firstTiling;
		
		public TilingsLayout(double [] w, int n, int ft){
			this.widths = w;
			this.numTilings = n;
			this.firstTiling = ft;
		}
		
		public double [] getWidths(){
			return this.widths;
		}
		
		public int getNumTilings(){
			return this.numTilings;
		}
		
		public int getFirstTiling(){
			return this.firstTiling;
		}
	}
	
	

}