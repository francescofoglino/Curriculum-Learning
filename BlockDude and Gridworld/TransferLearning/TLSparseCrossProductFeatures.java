package TransferLearning;

import burlap.behavior.functionapproximation.sparse.SparseStateActionFeatures;
import burlap.behavior.functionapproximation.sparse.SparseStateFeatures;
import burlap.behavior.functionapproximation.sparse.StateFeature;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A {@link SparseStateActionFeatures} implementation that takes as input a {@link SparseStateFeatures} object,
 * and turns it into state-action features taking the cross product of the features with the action set. This
 * implementation is lazy and creates state-action features as they are queried. Consequently, the state-action feature indices
 * for an action may not be consecutive.
 * @author James MacGlashan.
 */
public class TLSparseCrossProductFeatures implements SparseStateActionFeatures{

	protected SparseStateFeatures 									sFeatures;
	
	protected Map<Action, FeaturesMap> 								actionFeatures = new HashMap<Action, FeaturesMap>();
	
	protected int 													nextFeatureId = 0;
	
	private HashMap<Integer, ArrayList<Integer>> 					sfNsaf = new HashMap<Integer, ArrayList<Integer>>();

	
	public TLSparseCrossProductFeatures(SparseStateFeatures sFeatures) {
		this.sFeatures = sFeatures;
	}

	protected TLSparseCrossProductFeatures(SparseStateFeatures sFeatures, Map<Action, FeaturesMap> actionFeatures, int nextFeatureId, HashMap<Integer, ArrayList<Integer>> sfNsaf) {
		this.sFeatures = sFeatures;
		this.actionFeatures = actionFeatures;
		this.nextFeatureId = nextFeatureId;
		this.sfNsaf = sfNsaf;
	}

	@Override
	public List<StateFeature> features(State s, Action a) {
		List<StateFeature> sfs = sFeatures.features(s);
		List<StateFeature> safs = new ArrayList<StateFeature>(sfs.size());
		for(StateFeature sf : sfs){
			StateFeature saf;
			
			if(sf.id != -1) {
				saf = new StateFeature(actionFeature(a, sf.id), sf.value);
				
				if(this.sfNsaf.get(sf.id) != null){
					if(!this.sfNsaf.get(sf.id).contains(saf.id))
						this.sfNsaf.get(sf.id).add(saf.id);
				}else{
					this.sfNsaf.put(sf.id, new ArrayList<Integer>());
					this.sfNsaf.get(sf.id).add(saf.id);
				}
			}else {//in case of dummy statefeature
				saf = new StateFeature(-1, 0);
			}	
			safs.add(saf);				
		}
		return safs;
	}

	@Override
	public TLSparseCrossProductFeatures copy() {
		Map<Action, FeaturesMap> nfeatures = new HashMap<Action, FeaturesMap>(actionFeatures.size());
		for(Map.Entry<Action, FeaturesMap> e : actionFeatures.entrySet()){
			nfeatures.put(e.getKey(), e.getValue().copy());
		}
		return new TLSparseCrossProductFeatures(sFeatures.copy(), nfeatures, nextFeatureId, this.sfNsaf);
	}

	@Override
	public int numFeatures() {
		return this.sFeatures.numFeatures()*nextFeatureId;
	}

	protected int actionFeature(Action a, int from){
		FeaturesMap fmap = this.actionFeatures.get(a);
		if(fmap == null){
			fmap = new FeaturesMap();
			this.actionFeatures.put(a, fmap);
		}
		return fmap.getOrCreate(from);
	}
	
	public ArrayList<Integer> transferKnowledge(TLSparseCrossProductFeatures newTLSpCrPrFeatures, ArrayList<Integer> sfIDs) {
		ArrayList<Integer> allIndexToChange = new ArrayList<Integer>();
		
		Set<Action> allActions = this.actionFeatures.keySet();
		
		for(Action initA : allActions)
			newTLSpCrPrFeatures.actionFeatures.put(initA, newTLSpCrPrFeatures.new FeaturesMap());
		
		for(Integer sfID : sfIDs){
			allIndexToChange.addAll(this.sfNsaf.get(sfID));
			
			for(Action a : allActions){
				newTLSpCrPrFeatures.setActionFeature(a, sfID, this.actionFeatures);
				
			}
		}
		
		newTLSpCrPrFeatures.setNextFeatureId(this.nextFeatureId);
		
//		newTLSpCrPrFeatures.setAllActionFeatures(this.actionFeatures);
//		ArrayList<Integer> allIndexToChange2 = new ArrayList<Integer>();
//		for(Integer temp : this.sfNsaf.keySet())
//			allIndexToChange2.addAll(this.sfNsaf.get(temp));
			
		return allIndexToChange;
	}
	
	public void setSFeatures(SparseStateFeatures newVal){
		this.sFeatures = newVal;
	}
	
	public void setAllActionFeatures(Map<Action, FeaturesMap> newVal){
		this.actionFeatures = new HashMap<Action, FeaturesMap> (newVal);
	}
	
	public void setActionFeature(Action a, Integer sfID, Map<Action, FeaturesMap> sourceActionFeatures){
		Integer safID = new Integer(sourceActionFeatures.get(a).featuresMap.get(sfID));
		
		//FeaturesMap currActionFeature = sourceActionFeatures.get(a).copy();
		FeaturesMap currActionFeature = new FeaturesMap(new HashMap<Integer, Integer>(sourceActionFeatures.get(a).featuresMap));
		currActionFeature.put(sfID, safID);
		
		this.actionFeatures.put(a, currActionFeature);
	}
	
	public void setNextFeatureId(int newVal){
		this.nextFeatureId = newVal;
	}




	protected class FeaturesMap{
		Map<Integer, Integer> featuresMap = new HashMap<Integer, Integer>();

		public FeaturesMap() {
		}

		public FeaturesMap(Map<Integer, Integer> featuresMap) {
			this.featuresMap = featuresMap;
		}

		public void put(int from, int to){
			this.featuresMap.put(from, to);
		}

		public int getOrCreate(int from){
			Integer to = this.featuresMap.get(from);
			if(to == null){
				to = nextFeatureId++;
				this.featuresMap.put(from, to);
			}
			return to;
		}

		public FeaturesMap copy(){
			return new FeaturesMap(new HashMap<Integer, Integer>(featuresMap));
		}

	}
}
