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

public class TLDefineRangeNormalizedVarFeatGridWorld implements DenseStateFeatures{
	
	protected Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
	private int relFireNum = 0;
	private int relPitNum = 0;
	
	//connstructor
	public TLDefineRangeNormalizedVarFeatGridWorld(Map<Object, VariableDomain> domains) {
		this.domains = domains;
		
		for(int i = 0; i < this.domains.size(); ++i) {
			if(this.domains.containsKey("relFire" + i + ":x"))
				this.relFireNum++;
			
			if(this.domains.containsKey("relFire" + i + ":y"))
				this.relFireNum++;
			
			if(this.domains.containsKey("relPit" + i + ":x"))
				this.relPitNum++;
			
			if(this.domains.containsKey("relPit" + i + ":y"))
				this.relPitNum++;
		}	
	}
	
	@Override
	public double[] features(State s) {		
		boolean outOfRange = false;
		
		if(s instanceof GridWorldState){
			StateVariablesTranslator svt_fires = new StateVariablesTranslator("relFire", this.relFireNum);
			StateVariablesTranslator svt_pits = new StateVariablesTranslator("relPit", this.relPitNum);
			
			List<Double> vals = new ArrayList<Double>();
			
			List<Object> objKeys = s.variableKeys();
			ArrayList<String> strKeys = sortKeys(objKeys);
			
			//main loop for storing general variable domains and starting handling "relative" variables
			for(String key : strKeys){
				double d;
				
				if(key.contains("fire")) {//rel fire
					d = AgentDistanceFrom(key,s);
					svt_fires.put(key, d);
					
					continue;
				}else if(key.contains("pit")) {//rel pit
					d = AgentDistanceFrom(key,s);
					svt_pits.put(key, d);
					
					continue;
				}else {
					VariableDomain vd = this.domains.get(key);
					
					if(vd == null){
						System.err.println("Problem! domains do not contain any variable named: " + key);
						continue;
					}
					
					d = AgentDistanceFrom(key,s);
					
					double norm = vd.norm(d);
					
					if(norm < 0 || norm > 0.99){
						outOfRange = true;
						System.err.println("WARNING: variable - " + key + " - has value = " + d + " AKA outside the pre-defined interval ( " + vd.lower + " " + vd.upper + " )");
					}	
					
					vals.add(norm);
				}
			}
			
			//************************************* REL VAR MANAGEMENT *********************************
			//these two loops manage relative variable domains for pits and fires. In these cases norm 
			//can be 1, which represents a variable out of range
			for(String relKey : sortKeys(Arrays.asList(svt_fires.returnSVTranslation().keySet().toArray()))) {//rearrangement of fire variables
				VariableDomain vd = this.domains.get(relKey);
				
				if(vd == null){
					System.err.println("Problem! domains do not contain any variable named: " + relKey);
					continue;
				}
				
				double d = svt_fires.returnSVTranslation().get(relKey);
				
				double norm = vd.norm(d);
				
				if(norm < 0 || norm > 1)
					System.err.println("WARNING: variable - " + relKey + " - has value = " + d + " AKA outside the pre-defined interval ( " + vd.lower + " " + vd.upper + " )");
				
				vals.add(norm);
			}
			
			for(String relKey : sortKeys(Arrays.asList(svt_pits.returnSVTranslation().keySet().toArray()))) {//rearrangement of pit variables
				VariableDomain vd = this.domains.get(relKey);
				
				if(vd == null){
					System.err.println("Problem! domains do not contain any variable named: " + relKey);
					continue;
				}
				
				double d = svt_pits.returnSVTranslation().get(relKey);
				
				double norm = vd.norm(d);
				
				if(norm < 0 || norm > 1)
					System.err.println("WARNING: variable - " + relKey + " - has value = " + d + " AKA outside the pre-defined interval ( " + vd.lower + " " + vd.upper + " )");
				
				vals.add(norm);
			}
			//**********************************************************************************************
			
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
	
	//treasure, fire1, fire2, fire3, ... ,pitn, pitn-1, ...
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
			}else if(keyToString.contains("relFire")){//fire and relFire are exclusive, if one is present the other is not
				String[] parts = keyToString.split(":");
				int fireNumber = Integer.parseInt(parts[0].split("ire")[1]);//get the number
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
		    }else if(keyToString.contains("relPit")){//fire and relFire are exclusive, if one is present the other is not
				String[] parts = keyToString.split(":");
				int pitNumber = Integer.parseInt(parts[0].split("t")[1]);//get the number

				if(parts[1].equals("x")){
					allKeysToStrings.put(keys.size() - (2*pitNumber), keyToString);
				}else if(parts[1].equals("y")){
					allKeysToStrings.put(keys.size() - (2*pitNumber - 1), keyToString);
				}
				
			}
		}
		
		Set<Integer> intKeys = allKeysToStrings.keySet();
		
		ArrayList<String> keysStrings = new ArrayList<String>();
		for(int k : intKeys)
			keysStrings.add(allKeysToStrings.get(k));
		
		return keysStrings;
		
	}
	
	@Override
	public DenseStateFeatures copy() {
		return new TLDefineRangeNormalizedVarFeatGridWorld(new HashMap<Object, VariableDomain>(domains));
	}
	
	/** This function is crucial for allowing the transfer since it can pass the set of the keys to the upper  */
	public ArrayList<String> getSortedKeys(){
		return sortKeys(Arrays.asList(this.domains.keySet().toArray()));
	}
	
	private class StateVariablesTranslator{
		HashMap<String, SVinfo> allSVIs = new HashMap<String, SVinfo>();
		private int d = 2;//tolerance distance for sensors
		private int vdNum = 0;
		String relativeVDname = "";
		
		StateVariablesTranslator(String newName, int num){
			this.relativeVDname = newName;
			this.vdNum = num;
		}
		
		public void put(String fullKey, double val) {
//			if(val > d)
//				System.err.println("ERROR: your sensor range is " + d + " greater than the read distance " + val);
			
			String[] splittedKey = fullKey.split(":");
			
			if(allSVIs.keySet().contains(splittedKey[0])) {
				if(splittedKey[1].equals("x")) {
					allSVIs.get(splittedKey[0]).dx = val;
				}else	
					allSVIs.get(splittedKey[0]).dy = val;
			}else {
				if(splittedKey[1].equals("x")) {
					allSVIs.put(splittedKey[0], new SVinfo(val, 0));
				}else	
					allSVIs.put(splittedKey[0], new SVinfo(0, val));
			}
			
			//vdNum++;
		}
		
		public HashMap<String,Integer> returnSVTranslation() {
			int i = 1;
			HashMap<String,Integer> vd = new HashMap<String,Integer>();
			
			for(int y = -d; y<=d; ++y) {
				if(y <= 0) {
					for(int x = -(y+d); x<=(y+d); ++x) {
						if(allSVIs.containsValue(new SVinfo(x,y))) {
							vd.put(relativeVDname + i + ":x", x);
							vd.put(relativeVDname + i + ":y", y);
							
							++i;
						}
					}
				}else {
					for(int x = (y-d); x<= -(y-d); ++x) {
						if(allSVIs.containsValue(new SVinfo(x,y))) {
							vd.put(relativeVDname + i + ":x", x);
							vd.put(relativeVDname + i + ":y", y);
							
							++i;
						}
					}
				}	
			}
			
			if(vd.size() < vdNum) {
				for(; vd.size()<vdNum; ++i) {//dummy vals to get these varss out of the defined interval
					vd.put(relativeVDname + i + ":x", d+1);
					vd.put(relativeVDname + i + ":y", d+1);
				}
			}else if(vd.size() > vdNum)
				System.err.println("ERROR: too many artificial variable domains!");
			
			return vd;
		}
		
		public class SVinfo{
			public double dx;
			public double dy;
			
			SVinfo(double x, double y){
				dx = x;
				dy = y;
			}
			
			@Override
			public boolean equals(Object toBeComparedTo) {
				if(toBeComparedTo instanceof SVinfo) {
					if(this.dx == ((SVinfo)toBeComparedTo).dx && this.dy == ((SVinfo)toBeComparedTo).dy)
						return true;
					else 
						return false;
				}else
					return false;
			}
		}
	}
}
