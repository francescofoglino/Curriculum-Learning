package myProj;

import burlap.behavior.functionapproximation.dense.*;
import burlap.mdp.core.state.State;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A {@link DenseStateFeatures} that iterates through all state variables in a state
 * and places them into the returned double array. Note that the values for the state variable must
 * return a {@link Number} value. Alternatively, you may only have a subset of the state variables be used
 * by setting up a white list of the variables to use with the {@link #addToWhiteList(Object)} method.
 * If you do not add any variables to the white list, then it will be assumed that all variables should be used.
 * @author James MacGlashan.
 */
public class NumericVariableAndBooleanFeatures implements DenseStateFeatures {

	protected List<Object> whiteList = null;

	public NumericVariableAndBooleanFeatures() {
	}

	public NumericVariableAndBooleanFeatures(Object...keys) {
		this.whiteList = Arrays.asList(keys);
	}

	public NumericVariableAndBooleanFeatures(List<Object> whiteList) {
		this.whiteList = whiteList;
	}

	public NumericVariableAndBooleanFeatures addToWhiteList(Object variableKey){
		if(whiteList == null){
			this.whiteList = new ArrayList<Object>();
		}
		this.whiteList.add(variableKey);

		return this;
	}



	@Override
	public double[] features(State s) {

		if(this.whiteList == null){
			//then use all
			List<Object> keys = s.variableKeys();
			double [] vals = new double[keys.size()];
			int i = 0;
			for(Object key : keys){
				Object value = s.get(key);
				if (value instanceof Number) {
					Number n = (Number)value;
					vals[i] = n.doubleValue();
					i++;
				}else{
					Boolean b = (Boolean)value;
					vals[i] = (b== Boolean.TRUE)? 1:0;
					i++;
				}	
			}

			return vals;
		}

		//otherwise use white list
		double [] vals = new double[this.whiteList.size()];
		int i = 0;
		for(Object key : this.whiteList){
			Object value = s.get(key);
			if (value instanceof Number) {
				Number n = (Number)value;
				vals[i] = n.doubleValue();
				i++;
			}else{
				Boolean b = (Boolean)value;
				vals[i] = (b== Boolean.TRUE)? 1:0;
				i++;
			}	
		}

		return vals;
	}

	@Override
	public NumericVariableAndBooleanFeatures copy() {
		return new NumericVariableAndBooleanFeatures(new ArrayList<Object>(this.whiteList));
	}
}
