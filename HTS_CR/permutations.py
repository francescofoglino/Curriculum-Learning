import sys
import itertools

def perm(n, k):
	print(list(itertools.permutations(n, k)))
	
if __name__ == "__main__":

	n = list(sys.argv[1].replace('[','').replace(']','').split(','))
	k = int(sys.argv[2])
	
	perm(n, k);
