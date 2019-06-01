package TransferLearning;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.DifferentiableStateValue;
import burlap.behavior.functionapproximation.FunctionGradient;
import burlap.behavior.functionapproximation.sparse.SparseCrossProductFeatures;
import burlap.behavior.functionapproximation.sparse.SparseStateFeatures;
import burlap.behavior.functionapproximation.sparse.StateFeature;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TLLinearVFA implements DifferentiableStateValue, DifferentiableStateActionValue{

	/**
	 * The state features
	 */
	public SparseStateFeatures 						sparseStateFeatures;

	/**
	 * The State-action features based on the cross product of state features and actions
	 */
	protected TLSparseCrossProductFeatures 			stateActionFeatures;
	
	/**
	 * A map from feature identifiers to function weights
	 */
	protected Map<Integer, Double>					weights;
	
	/**
	 * A default weight for the functions
	 */
	protected double								defaultWeight = 0.0;




	protected List<StateFeature>					currentFeatures;
	protected double								currentValue;
	protected FunctionGradient 						currentGradient = null;

	protected State									lastState = null;
	protected Action 								lastAction = null;


	/**
	 * Initializes with a feature database; the default weight value will be zero
	 * @param sparseStateFeatures the feature database to use
	 */
	public TLLinearVFA(SparseStateFeatures sparseStateFeatures) {

		this.sparseStateFeatures = sparseStateFeatures;
		this.stateActionFeatures = new TLSparseCrossProductFeatures(sparseStateFeatures);
		this.weights = new HashMap<Integer, Double>();


	}


	/**
	 * Initializes
	 * @param sparseStateFeatures the feature database to use
	 * @param defaultWeight the default feature weight to initialize feature weights to
	 */
	public TLLinearVFA(SparseStateFeatures sparseStateFeatures, double defaultWeight) {

		this.sparseStateFeatures = sparseStateFeatures;
		this.stateActionFeatures = new TLSparseCrossProductFeatures(sparseStateFeatures);
		this.defaultWeight = defaultWeight;
		this.weights = new HashMap<Integer, Double>();


	}




	@Override
	public double evaluate(State s, Action a) {

		List<StateFeature> features = this.stateActionFeatures.features(s, a);
		double val = 0.;
		for(StateFeature sf : features){
			double prod = sf.value * this.getWeight(sf.id);
			val += prod;
		}
		this.currentValue = val;
		this.currentGradient = null;
		this.currentFeatures = features;
		this.lastState = s;
		this.lastAction = a;
		return val;
	}

	@Override
	public double evaluate(State s) {
		List<StateFeature> features = this.sparseStateFeatures.features(s);
		double val = 0.;
		for(StateFeature sf : features){
			double prod = sf.value * this.getWeight(sf.id);
			val += prod;
		}
		this.currentValue = val;
		this.currentGradient = null;
		this.currentFeatures = features;
		this.lastState = s;
		this.lastAction = null;
		return this.currentValue;
	}


	@Override
	public FunctionGradient gradient(State s) {

		List<StateFeature> features;

		if(this.lastState == s && this.lastAction == null){
			if(this.currentGradient != null) {
				return this.currentGradient;
			}
			features = this.currentFeatures;
		}
		else{
			features = this.sparseStateFeatures.features(s);
		}

		FunctionGradient gd = new FunctionGradient.SparseGradient(features.size());
		for(StateFeature sf : features){
			gd.put(sf.id, sf.value);
		}
		this.currentGradient = gd;
		this.lastState = s;
		this.lastAction = null;
		this.currentFeatures = features;

		return gd;
	}

	@Override
	public FunctionGradient gradient(State s, Action a) {

		List<StateFeature> features;

		if(this.lastState == s && this.lastAction == a){
			if(this.currentGradient != null) {
				return this.currentGradient;
			}
			features = this.currentFeatures;
		}
		else{
			features = this.stateActionFeatures.features(s, a);
		}

		FunctionGradient gd = new FunctionGradient.SparseGradient(features.size());
		for(StateFeature sf : features){
			gd.put(sf.id, sf.value);
		}
		this.currentGradient = gd;
		this.lastState = s;
		this.lastAction = a;
		this.currentFeatures = features;

		return gd;
	}


	@Override
	public int numParameters() {
		return this.weights.size();
	}

	@Override
	public double getParameter(int i) {
		return this.getWeight(i);
	}

	@Override
	public void setParameter(int i, double p) {
		this.weights.put(i, p);
	}

	protected double getWeight(int weightId){
		Double stored;
		
		if(weightId != -1) {
			stored = this.weights.get(weightId);
			if(stored == null){
				this.weights.put(weightId, this.defaultWeight);
				return this.defaultWeight;
			}
		}else
			stored = 0.;
		
		return stored;
	}


	@Override
	public void resetParameters() {
		this.weights.clear();
	}

	@Override
	public TLLinearVFA copy() {

		TLLinearVFA vfa = new TLLinearVFA(this.sparseStateFeatures.copy(), this.defaultWeight);
		vfa.weights = new HashMap<Integer, Double>(this.weights.size());
		vfa.stateActionFeatures = stateActionFeatures.copy();
		for(Map.Entry<Integer, Double> e : this.weights.entrySet()){
			vfa.weights.put(e.getKey(), e.getValue());
		}

		return vfa;
	}
	
	/** Transfer Learning stuff 
	 *  @param maskForSelection has the exact shape of the tilings one wants to select or, in case all the tilings are needed indistinctly, it is set to null
	 * */
	public void transferVFA(TLLinearVFA newTLLVFA, boolean [] maskForSelection){
		
		if(this.sparseStateFeatures instanceof TLTileCodingFeatures && newTLLVFA.sparseStateFeatures instanceof TLTileCodingFeatures){
			
			/****************** Transfer State Features ******************/
			TLTileCodingFeatures sourceFeatures = (TLTileCodingFeatures)this.sparseStateFeatures;
			TLTileCodingFeatures targetFeatures = (TLTileCodingFeatures)newTLLVFA.sparseStateFeatures;
			
			ArrayList<Integer> indexToChangeSF = new ArrayList<Integer> (sourceFeatures.transferKnowledge(targetFeatures, maskForSelection));
			/*************************** END *****************************/
			
			/*************** Transfer State Action Features **************/
			TLSparseCrossProductFeatures sourceActionFeatures = this.stateActionFeatures;
			TLSparseCrossProductFeatures targetActionFeatures = newTLLVFA.stateActionFeatures;
			
			ArrayList<Integer> indexToChangeSAF = sourceActionFeatures.transferKnowledge(targetActionFeatures, indexToChangeSF);
			/*************************** END *****************************/
			
			//Now that all the parameters are correctly selected we can transfer their value one by one
			for(int i = 0; i<this.numParameters(); ++i){
				if(indexToChangeSAF.contains(i))
					newTLLVFA.setParameter(i, this.getParameter(i));
			}
			
		}else{
			System.out.println("ERROR: You are trying to perform transfer without using the ad hoc implemented classes");
		}
		
		return;
	}

}
