package org.algo.sorting;

import java.util.Arrays;

/*
 * Use partition to divide and conquer. Divide sample set into small partitions based on a pivot such that
 * elements on left of pivot are less than pivot and elements on right of pivot are greater than the pivot.
 * 
 *  Logic to partition an array.
 *  Method signature partition(Array, startIndex, endIndex)
 *  Take any element as pivot lets say , we always take the right most element as the pivot 
 *  start comparing the pivot with the array elements such that
 *  pivot <- A[end]
 *  partitionIndex <- start
 *  for i <- start to end-1 *  {
 *  	if(A[i] <= pivot){
 *  		swap(A[i],A[partionIndex])
 *  		partitionIndex++
 *  	}
 *  		
 *  
 */

public class QuickSort {

	public static void main(String[] args) {
		int[] array = {15,3,18,2,14,9,12,4};
		quickSort(array,0,array.length-1);

	}
	
	private static void quickSort(int[] array, int start, int end){
		if(start < end){
			int partition = partition(array,start, end);
			quickSort(array,start,partition-1);
			quickSort(array,partition+1,end);
		}
	}
	
	private static int partition(int[] array,int start,int end){
		int pIndex = start;
		int pivot = array[end];
		System.out.println("Pivot = " + pivot + " Start " + start + " end " + end );
		for(int i=start;i < end;i++){
			if(array[i] <= pivot){
				swap(array, i,pIndex);
				pIndex++;
			}
			System.out.println(Arrays.toString(array));
		}
		swap(array,pIndex,end);
		System.out.println(Arrays.toString(array));
		return pIndex;
	}
	
	private static void swap(int[] array, int a, int b){
		int tmp = array[a];
		array[a] = array[b];
		array[b] = tmp;
	}

}
