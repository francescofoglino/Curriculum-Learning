package TransferLearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

//import org.hamcrest.core.IsInstanceOf;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.domain.singleagent.blockdude.state.BlockDudeMap;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;

public class TLNormalizedVarFeatBlockDude implements DenseStateFeatures{
	
	protected Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
	
	//the only constructor
	public TLNormalizedVarFeatBlockDude(Map<Object, VariableDomain> domains) {
		this.domains = domains;
	}
	
	@Override
	public double[] features(State s) {		
		boolean outOfRange = false;
		
		if(s instanceof BlockDudeState){
			List<Double> vals = new ArrayList<Double>();
			
			List<Object> objKeys = s.variableKeys();
			ArrayList<String> strKeys = sortKeys(objKeys);
			for(String key : strKeys){
				VariableDomain vd = this.domains.get(key);
				double d;
				if(vd == null){
					
					if(this.domains.get("column1:x") == null)//trick to avoid this loop in case we have no columns
						continue;
					
					//here we handle the columns
					if(key.contains("map")){
						int [][] map = (int[][]) s.get(key);
						int r = 0;
						ArrayList<Integer[]> columns = new ArrayList<Integer[]>();
						
						//boolean leftBlock = true;//for checking if there is a block on the left of the current column
						
						for(int c = 1; c < map.length - 1; ++c){//excluding map margins
							r = columnHeight(map[c]); 
							if(map[c-1][r] != 1 || map[c+1][r] != 1)
								columns.add(new Integer []{r,c});
						}
						
						if(!columns.isEmpty()){
							String newKey = "";
							for(Integer str=1; str <= columns.size(); ++str){
								newKey = "column";
								newKey = newKey + str.toString();
								
								Integer [] rc = columns.get(str-1);
								int y = rc[0], x = rc[1];
								
								double norm;
								
								d = x - ((Number)(s.get("agent:x"))).doubleValue();
								
								vd = this.domains.get(newKey+":x");
								norm = vd.norm(d);
								vals.add(norm);
								
								d = y - ((Number)(s.get("agent:y"))).doubleValue();
								
								vd = this.domains.get(newKey+":y");
								norm = vd.norm(d);
								vals.add(norm);
							}	
						}	
					}
					continue;
				}
				Object value = s.get(key);
				if(value instanceof Number){
					if(key.contains("block")){
						d = AgentDistanceFrom(key,s);
					}else if(key.contains("exit")){
						d = AgentDistanceFrom(key,s);
					}else{
						d = ((Number)value).doubleValue();
					}
					double norm = vd.norm(d);
					
					if(norm < 0 || norm > 1){
						outOfRange = true;
						System.err.println("WARNING: variable - " + key + " - has value = " + d + " AKA outside the pre-defined interval ( " + vd.lower + " " + vd.upper + " )");
					}	
					
					vals.add(norm);
				}else{
					Boolean b = (Boolean)value;
					double conv = (b== Boolean.TRUE)? 1:0;
					double norm = vd.norm(conv);
					vals.add(norm);
				}
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
			System.out.println("The input State must be a BlockDudeState");
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
	
	private int columnHeight(int [] column){
		int height = -1;//dummy value
		for(int i = 0; i<column.length; ++i){
//			if(column[i]==0){//this version counts columns as a succession of 1s from heigth 0 until the first zero encountered
//				height = i-1;
//				break;
//			}
			if(column[i] == 1)//this version counts columns as the heighest 1 encountered
				height = i;
		}
		return height;
	}

	private ArrayList<String> sortKeys(List<Object> keys){
		
		HashMap<Integer,String> allKeysToStrings = new HashMap<Integer,String>();
		
		for(Object key : keys){
			String keyToString = key.toString();
			if(keyToString.contains("agent")){
				if(keyToString.contains("dir"))
					allKeysToStrings.put(0, keyToString);//always the first element
				else if(keyToString.contains("holding"))
					allKeysToStrings.put(1, keyToString);
				
			}else if(keyToString.contains("exit")){
				if(keyToString.contains(":x"))
					allKeysToStrings.put(2, keyToString);
				else if(keyToString.contains(":y"))
					allKeysToStrings.put(3, keyToString);
				
			}else if(keyToString.contains("block")){
				String[] parts = keyToString.split(":");
				int blockNumber = Integer.parseInt(parts[0].split("k")[1]);//get the number
				int partialIndex = 3 + (2*blockNumber);
				if(parts[1].equals("x")){
					allKeysToStrings.put(partialIndex - 1, keyToString);
				}else if(parts[1].equals("y")){
					allKeysToStrings.put(partialIndex, keyToString);
				}
			//when we have the key map we also have agent:x and agent:y. This case corresponds
			//to the call in the function features. It justifies the -3. Also when map is there
			//then we don't have column and vice versa	
			}else if(keyToString.contains("map")){
				allKeysToStrings.put((keys.size() - 3), keyToString);
				
			}else if(keyToString.contains("column")){
				String[] parts = keyToString.split(":");
				int columnNumber = Integer.parseInt(parts[0].split("n")[1]);//get the number

				if(parts[1].equals("x")){
					allKeysToStrings.put(keys.size() - (2*columnNumber), keyToString);
				}else if(parts[1].equals("y")){
					allKeysToStrings.put(keys.size() - (2*columnNumber - 1), keyToString);
				}
				
			}
		}
		
		/** Array of incremental values from 0 to size() */
		int[] intKeys = new int[allKeysToStrings.size()];
		for(int i = 0; i < allKeysToStrings.size(); ++i){
			intKeys[i] = i;
		}
		
		ArrayList<String> keysStrings = new ArrayList<String>();
		for(int k : intKeys)
			keysStrings.add(allKeysToStrings.get(k));
		
		return keysStrings;
		
	}
	
	@Override
	public DenseStateFeatures copy() {
		return new TLNormalizedVarFeatBlockDude(new HashMap<Object, VariableDomain>(domains));
	}
	
	/** This function is crucial for allowing the transfer since it can pass the set of the keys to the upper  */
	public ArrayList<String> getSortedKeys(){
		return sortKeys(Arrays.asList(this.domains.keySet().toArray()));
	}
}
