package myProj;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

//import org.hamcrest.core.IsInstanceOf;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;

public class NormalizedVariableFeaturesBlockDude implements DenseStateFeatures{
	
	protected Map<Object, VariableDomain> domains = new HashMap<Object, VariableDomain>();
	
	//the only constructor
	public NormalizedVariableFeaturesBlockDude(Map<Object, VariableDomain> domains) {
		this.domains = domains;
	}
	
	@Override
	public double[] features(State s) {		
		boolean outOfRange = false;
		
		if(s instanceof BlockDudeState){
			List<Double> vals = new ArrayList<Double>();
			
			List<Object> keys = s.variableKeys();
			for(Object key : keys){
				VariableDomain vd = this.domains.get(key.toString());
				if(vd == null){
					continue;
				}
				Object value = s.get(key);
				if(value instanceof Number){
					double d;
					if(key.toString().contains("block")){
						d = AgentDistanceFrom(key,s);
					}else if(key.toString().contains("exit")){
						d = AgentDistanceFrom(key,s);
					}else{
						d = ((Number)value).doubleValue();
					}
					double norm = vd.norm(d);
					
					if(norm < 0 || norm > 1)
						outOfRange = true;
					
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
			if(outOfRange)
				return valsToArray;
			
			for(int i=0; i<vals.size(); ++i){
				valsToArray[i] = vals.get(i).doubleValue();
			}
			return valsToArray;
		}else{
			System.out.println("The input State must be a BlockDudeState");
			return null;
		}	
	}
	
	private double AgentDistanceFrom(Object key, State state){
		String keyString = key.toString();
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

	@Override
	public DenseStateFeatures copy() {
		return new NormalizedVariableFeaturesBlockDude(new HashMap<Object, VariableDomain>(domains));
	}
}
