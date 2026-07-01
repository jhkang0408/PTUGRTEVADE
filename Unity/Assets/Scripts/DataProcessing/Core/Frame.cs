#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

[System.Serializable]
public class Frame {
	public MotionData Data;
	public int Index;
	public float Timestamp;
	public Matrix4x4 Head;
	public Vector3 GazeOrigin;
	public Vector3 GazeDirection;
	public Matrix4x4 RightHand;
	public Vector3 RightHand_IndexFingerTip;
	public Matrix4x4 LeftHand;
	public Vector3 LeftHand_IndexFingerTip;
	public Vector3[] Targets = new Vector3[3];

	public Frame(MotionData data, int index, float timestamp) 
	{
		Data = data;
		Index = index;
		Timestamp = timestamp;
	}

	public Frame GetFirstFrame() 
	{
		return Data.Frames[0];
	}

	public Frame GetLastFrame() 
	{
		return Data.Frames[Data.Frames.Length-1];
	}

	public Matrix4x4 Get_Head()
	{		
		return Head;
	}

	public Vector3 Get_GazeOrigin()
	{
		return GazeOrigin;
	}

	public Vector3 Get_GazeDirection()
	{
		return GazeDirection;
	} 

	public Matrix4x4 Get_RightHand()
	{
		return RightHand;
	}

	public Vector3 Get_RightHand_IndexFingerTip()
    {
		return RightHand_IndexFingerTip;
    }

	public Matrix4x4 Get_LeftHand()
	{
		return LeftHand;
	}

	public Vector3 Get_LeftHand_IndexFingerTip()
	{
		return LeftHand_IndexFingerTip;
	} 

	public Vector3 Get_Target(int index)
    {
		return Targets[index];
    }
}
#endif