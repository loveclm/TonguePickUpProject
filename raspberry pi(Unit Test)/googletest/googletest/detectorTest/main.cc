#include "detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/**
* main function
* usage: detector filepath
* @param filepath filepath to be processed
* @author RG Jong
* @date 2017/8/1
*/


int main(int argc, char* argv[]){

	if(argc == 2){
		
		tongueDetectionAlgorithmUpgrade(argv[1], 0);
		return 0;
		
	} else {
		printf("Usages :\n\t%s filename\n", argv[0]);		
	}
	return 0;
	
}


