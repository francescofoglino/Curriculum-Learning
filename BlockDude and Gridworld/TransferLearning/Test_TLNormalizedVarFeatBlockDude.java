package TransferLearning;

import BlockDude.*;

import static org.junit.Assert.assertArrayEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import burlap.mdp.core.state.vardomain.VariableDomain;
import junit.framework.TestCase;

public class Test_TLNormalizedVarFeatBlockDude extends TestCase {

	@Test
	public void test() {
		
		/** TEST MAP_ 1 
		 * 4 basic state variables*/
		
		Map<Object, VariableDomain> domains_M1 = new HashMap<Object, VariableDomain>();
		domains_M1.put("exit:x", new VariableDomain(0,7));
		domains_M1.put("exit:y", new VariableDomain(-3,0));// bigger than what it should be
		domains_M1.put("agent:dir",new VariableDomain(0,1));
		domains_M1.put("agent:holding",new VariableDomain(0,1));
		
		TLNormalizedVarFeatBlockDude stateVariables_M1 = new TLNormalizedVarFeatBlockDude(domains_M1);
		
		double[] exp_M1 = {0, 0, 1, 1};
		double[] out_M1 = stateVariables_M1.features(BlockDudeTransferLearning_UTILS.initialState_1);
		
		System.out.println("TEST MAP_ 1\n");
		for(double d : out_M1)
			System.out.println(d);
		System.out.println("END \n");
		
		assertArrayEquals(exp_M1, out_M1, 0.00001);
		
		/** TEST MAP_ 2 
		 * 4 basic state variables
		 * 2 state vars for the column*/
		
		Map<Object, VariableDomain> domains_M2 = new HashMap<Object, VariableDomain>();
		domains_M2.put("exit:x", new VariableDomain(0,7));
		domains_M2.put("exit:y", new VariableDomain(-3,0));// bigger than what it should be
		domains_M2.put("agent:dir",new VariableDomain(0,1));
		domains_M2.put("agent:holding",new VariableDomain(0,1));
		domains_M2.put("column1:x", new VariableDomain(-3,4));
		domains_M2.put("column1:y", new VariableDomain(-1,0));
		
		TLNormalizedVarFeatBlockDude stateVariables_M2 = new TLNormalizedVarFeatBlockDude(domains_M2);
		
		double[] exp_M2 = {0, 0, 1, 1, 1, 1};
		double[] out_M2 = stateVariables_M2.features(BlockDudeTransferLearning_UTILS.initialState_2);
		
		System.out.println("TEST MAP_ 2\n");
		for(double d : out_M2)
			System.out.println(d);
		System.out.println("END \n");
		
		assertArrayEquals(exp_M2, out_M2, 0.00001);
		
		/** TEST MAP_ 3  
		 * 4 basic state variables
		 * 2 state vars for the block*/
		
		Map<Object, VariableDomain> domains_M3 = new HashMap<Object, VariableDomain>();
		domains_M3.put("exit:x", new VariableDomain(0,7));
		domains_M3.put("exit:y", new VariableDomain(-3,0));// bigger than what it should be
		domains_M3.put("agent:dir",new VariableDomain(0,1));
		domains_M3.put("agent:holding",new VariableDomain(0,1));
		domains_M3.put("block1:x", new VariableDomain(-4,4));
		domains_M3.put("block1:y", new VariableDomain(-2,1));
		
		TLNormalizedVarFeatBlockDude stateVariables_M3 = new TLNormalizedVarFeatBlockDude(domains_M3);
		
		double[] exp_M3 = {0, 0, 1, 1, 7./8, 2./3};
		double[] out_M3 = stateVariables_M3.features(BlockDudeTransferLearning_UTILS.initialState_3);
		
		System.out.println("TEST MAP_ 3\n");
		for(double d : out_M3)
			System.out.println(d);
		System.out.println("END \n");
		
		assertArrayEquals(exp_M3, out_M3, 0.00001);
		
		/** TEST MAP_ 3 #2
		 * 4 basic state variables
		 * 2 state vars for the column
		 * 2 state vars for the block*/
		
		Map<Object, VariableDomain> domains_M3_2 = new HashMap<Object, VariableDomain>();
		domains_M3_2.put("exit:x", new VariableDomain(0,7));
		domains_M3_2.put("exit:y", new VariableDomain(-3,0));// bigger than what it should be
		domains_M3_2.put("agent:dir",new VariableDomain(0,1));
		domains_M3_2.put("agent:holding",new VariableDomain(0,1));
		domains_M3_2.put("block1:x", new VariableDomain(-4,4));
		domains_M3_2.put("block1:y", new VariableDomain(-2,1));
		domains_M3_2.put("column1:x", new VariableDomain(-2,5));
		domains_M3_2.put("column1:y", new VariableDomain(-1,1));
		
		TLNormalizedVarFeatBlockDude stateVariables_M3_2 = new TLNormalizedVarFeatBlockDude(domains_M3_2);
		
		double[] exp_M3_2 = {0, 0, 1, 1, 7./8, 2./3, 1, 1};
		double[] out_M3_2 = stateVariables_M3_2.features(BlockDudeTransferLearning_UTILS.initialState_3);
		
		System.out.println("TEST MAP_ 3 #2\n");
		for(double d : out_M3_2)
			System.out.println(d);
		System.out.println("END \n");
		
		assertArrayEquals(exp_M3_2, out_M3_2, 0.00001);
		
		/** TEST MAP_ 3  
		 * 4 basic state variables
		 * 2 state vars for the block*/
		
		Map<Object, VariableDomain> domains_M4 = new HashMap<Object, VariableDomain>();
		domains_M4.put("exit:x", new VariableDomain(0,7));
		domains_M4.put("exit:y", new VariableDomain(-3,0));// bigger than what it should be
		domains_M4.put("agent:dir",new VariableDomain(0,1));
		domains_M4.put("agent:holding",new VariableDomain(0,1));
		domains_M4.put("block1:x", new VariableDomain(-4,4));
		domains_M4.put("block1:y", new VariableDomain(-3,1));
		domains_M4.put("block2:x", new VariableDomain(-4,4));
		domains_M4.put("block2:y", new VariableDomain(-3,1));
		domains_M4.put("block3:x", new VariableDomain(-4,4));
		domains_M4.put("block3:y", new VariableDomain(-3,1));
		
		TLNormalizedVarFeatBlockDude stateVariables_M4 = new TLNormalizedVarFeatBlockDude(domains_M4);
		
		double[] exp_M4 = {0, 0, 1, 1, 6./8, 3./4, 7./8, 3./4, 1, 3./4};
		double[] out_M4 = stateVariables_M4.features(BlockDudeTransferLearning_UTILS.initialState_4);
		
		System.out.println("TEST MAP_ 4\n");
		for(double d : out_M4)
			System.out.println(d);
		System.out.println("END \n");
		
		assertArrayEquals(exp_M4, out_M4, 0.00001);
		
	}

}
