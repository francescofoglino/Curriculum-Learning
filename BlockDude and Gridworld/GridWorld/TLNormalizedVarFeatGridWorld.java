package GridWorld;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;

public class TLNormalizedVarFeatGridWorld implements DenseStateFeatures{
	
	protected Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
	
	//connstructor
	public TLNormalizedVarFeatGridWorld(Map<Object, VariableDomain> domains) {
		this.domains = domains;
	}
	
	@Override
	public double[] features(State s) {		
		boolean outOfRange = false;
		
		if(s instanceof GridWorldState){
			List<Double> vals = new ArrayList<Double>();
			
			List<Object> objKeys = s.variableKeys();
			ArrayList<String> strKeys = sortKeys(objKeys);
			for(String key : strKeys){
				VariableDomain vd = this.domains.get(key);
				double d;
				
				if(vd == null){
					System.err.println("Problem! domains do not contain any variable named: " + key);
					continue;
				}
				
				d = AgentDistanceFrom(key,s);
				
				double norm = vd.norm(d);
				
				if(norm < 0 || norm > 1){
					outOfRange = true;
					System.err.println("WARNING: variable - " + key + " - has value = " + d + " AKA outside the pre-defined interval ( " + vd.lower + " " + vd.upper + " )");
				}	
				
				vals.add(norm);
				
			}
			double[] valsToArray = new double[vals.size()];
			
			//if any of the values is out of range the whole correspondent features are "killed"
			if(outOfRange){
				System.err.println("ERROR: one of the variables is out of its predefined range");
				return valsToArray;
			}	
			
			for(int i=0; i<vals.size(); ++i){
				valsToArray[i] = vals.get(i).doubleValue();
			}
			return valsToArray;
		}else{
			System.err.println("The input State must be a GridWorldState");
			return null;
		}	
	}
	
	//calculates the distance between the agent and the object taken into consideration
	private double AgentDistanceFrom(String key, State state){
		String keyString = key;
		double objectDoubleValue = ((Number)(state.get(keyString))).doubleValue();
		double agentDoubleValue;
		
		if(keyString.contains(":x")){
			agentDoubleValue = ((Number)(state.get("agent:x"))).doubleValue();			
		}else if(keyString.contains(":y")){
			agentDoubleValue = ((Number)(state.get("agent:y"))).doubleValue();
		}else{
			System.out.println("ERROR");
			return 0;
		}
		return objectDoubleValue - agentDoubleValue;
	}
	
	//treasure, fire1, fire2, fire3, ... ,pit1, pit2, ...
	private ArrayList<String> sortKeys(List<Object> keys){
		
		HashMap<Integer,String> allKeysToStrings = new HashMap<Integer,String>();
		
		for(Object key : keys){
			String keyToString = key.toString();
			if(keyToString.contains("treasure")){
				if(keyToString.contains(":x"))
					allKeysToStrings.put(0, keyToString);//always the first element
				else if(keyToString.contains(":y"))
					allKeysToStrings.put(1, keyToString);
				
			}else if(keyToString.contains("fire")){
				String[] parts = keyToString.split(":");
				int fireNumber = Integer.parseInt(parts[0].split("e")[1]);//get the number
				int partialIndex = 1 + (2*fireNumber);
				if(parts[1].equals("x")){
					allKeysToStrings.put(partialIndex - 1, keyToString);
				}else if(parts[1].equals("y")){
					allKeysToStrings.put(partialIndex, keyToString);
				}
			}else if(keyToString.contains("pit")){
				String[] parts = keyToString.split(":");
				int pitNumber = Integer.parseInt(parts[0].split("t")[1]);//get the number

				if(parts[1].equals("x")){
					allKeysToStrings.put(keys.size() - (2*pitNumber), keyToString);
				}else if(parts[1].equals("y")){
					allKeysToStrings.put(keys.size() - (2*pitNumber - 1), keyToString);
				}
				
			}
		}
		
		/** Array of incremental values from 0 to size() */
//		int[] intKeys = new int[allKeysToStrings.size()];
//		for(int i = 0; i < allKeysToStrings.size(); ++i){
//			intKeys[i] = i;
//		}
		
		Set<Integer> intKeys = allKeysToStrings.keySet();
		
		ArrayList<String> keysStrings = new ArrayList<String>();
		for(int k : intKeys)
			keysStrings.add(allKeysToStrings.get(k));
		
		return keysStrings;
		
	}
	
	@Override
	public DenseStateFeatures copy() {
		return new TLNormalizedVarFeatGridWorld(new HashMap<Object, VariableDomain>(domains));
	}
	
	/** This function is crucial for allowing the transfer since it can pass the set of the keys to the upper  */
	public ArrayList<String> getSortedKeys(){
		return sortKeys(Arrays.asList(this.domains.keySet().toArray()));
	}
}
