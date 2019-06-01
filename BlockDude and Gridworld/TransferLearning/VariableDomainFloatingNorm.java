package TransferLearning;

import burlap.mdp.core.state.vardomain.VariableDomain;

/**
 * A tuple specifying the numeric domain of a {@link burlap.mdp.core.state.State} variable.
 * Based on the code from @author James MacGlashan.
 * This version includes the possibility of specifying the interval on which one wants to normalize his values by setting @var newLower and @var newUpper. * 
 */
public class VariableDomainFloatingNorm extends VariableDomain{

	public double newLower = 0.;

	/**
	 * The upper value of the new domain interval
	 */
	public double newUpper = 0.99;
	
	/**
	 * Initializes.
	 * @param lower The lower value of the domain
	 * @param upper The upper value of the domain
	 */
	public VariableDomainFloatingNorm(double lower, double upper) {
		this.lower = lower;
		this.upper = upper;
	}

	/**
	 * Given a value in this variable domain, returns its normalized value. That is,
	 * (d - lower) / (upper - lower)
	 * @param d the input value
	 * @return the normalized value
	 */
	@Override
	public double norm(double d){
		double norm = newLower + ((d - lower)*(newUpper - newLower) / this.span());
		if(norm < newLower || norm > newUpper) {
			//System.out.println("LOOK! You are playing with fire here");
			norm = 1.;
		}
					
		return norm;
	}
}
