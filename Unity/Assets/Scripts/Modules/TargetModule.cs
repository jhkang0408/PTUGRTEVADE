#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

public class TargetModule : Module
{

	public bool[] Head_Keys = new bool[0];
	public bool[] RightHand_Keys = new bool[0];
	public bool[] LeftHand_Keys = new bool[0];
	public TargetFunction Head_Target = null;
	public TargetFunction RightHand_Target = null;
	public TargetFunction LeftHand_Target = null;
	public bool IsHeadTarget1 = false;
	public bool IsHeadTarget2 = false;
	public bool IsHeadTarget3 = false;
	public bool IsHeadTargetNull = false;
	public bool IsRightHandTarget1 = false;
	public bool IsRightHandTarget2 = false;
	public bool IsRightHandTarget3 = false;
	public bool IsRightHandTargetNull = false;
	public bool IsLeftHandTarget1 = false;
	public bool IsLeftHandTarget2 = false;
	public bool IsLeftHandTarget3 = false;
	public bool IsLeftHandTargetNull = false; 

	public override ID GetID()
	{
		return ID.Target;
	}

	public override Module Initialise(MotionData data)
	{
		Data = data;
		// Head
		Head_Target = new TargetFunction(this);		
		Head_Keys = new bool[data.GetTotalFrames()];
		Head_Keys[0] = true;
		Head_Keys[Head_Keys.Length - 1] = true;
		for (int i=0; i< data.GetTotalFrames(); i++)
        {
			Head_Target.Props[i] = "TargetNull";
		}		
		// RightHand
		RightHand_Target = new TargetFunction(this);
		RightHand_Keys = new bool[data.GetTotalFrames()];
		RightHand_Keys[0] = true;
		RightHand_Keys[RightHand_Keys.Length - 1] = true;
		for (int i = 0; i < data.GetTotalFrames(); i++)
		{
			RightHand_Target.Props[i] = "TargetNull";
		}
		// LeftHand
		LeftHand_Target = new TargetFunction(this);
		LeftHand_Keys = new bool[data.GetTotalFrames()];
		LeftHand_Keys[0] = true;
		LeftHand_Keys[LeftHand_Keys.Length - 1] = true;
		for (int i = 0; i < data.GetTotalFrames(); i++)
		{
			LeftHand_Target.Props[i] = "TargetNull";
		}
		return this;
	}

	public override void Slice(Sequence sequence)
	{
		Debug.Log("Slicing not implemented for " + GetID() + ".");
	}

	public override void Callback(MotionEditor editor)
	{

	}

	public void ToggleKey(TargetFunction Target, Frame frame, bool[] keys)
	{
		keys[frame.Index - 1] = !keys[frame.Index - 1];
		Target.Compute(frame, keys);
	}

	public bool IsKey(Frame frame, bool[] keys)
	{
		return keys[frame.Index - 1];
	}

	public Frame GetPreviousKey(Frame frame, bool[] keys)
	{
		while (frame.Index > 1)
		{
			frame = Data.GetFrame(frame.Index - 1);
			if (IsKey(frame, keys))
			{
				return frame;
			}
		}
		return Data.Frames.First();
	}

	public Frame GetNextKey(Frame frame, bool[] keys)
	{
		while (frame.Index < Data.GetTotalFrames())
		{
			frame = Data.GetFrame(frame.Index + 1);
			if (IsKey(frame, keys))
			{
				return frame;
			}
		}
		return Data.Frames.Last();
	}

	protected override void DerivedDraw(MotionEditor editor)
	{
	}

	protected override void DerivedInspector(MotionEditor editor)
	{
		Frame frame = editor.GetCurrentFrame();

        #region Head
        if (Utility.GUIButton("Key (Head)", IsKey(frame, Head_Keys) ? UltiDraw.Cyan : UltiDraw.DarkGrey, IsKey(frame, Head_Keys) ? UltiDraw.Black : UltiDraw.White))
		{
			ToggleKey(Head_Target,frame, Head_Keys);
		}

		EditorGUI.BeginDisabledGroup(!IsKey(frame, Head_Keys));

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("Head Target", GUILayout.Width(100f));
		EditorGUILayout.TextField(Head_Target.GetProp(frame));
		if (Utility.GUIButton("Target1", IsHeadTarget1 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsHeadTarget1 = !IsHeadTarget1;
			if (IsHeadTarget1)
            {
				Head_Target.SetProp(frame, "Target1", Head_Keys);
				IsHeadTarget1 = false;
			}			
		}
		if (Utility.GUIButton("Target2", IsHeadTarget2 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsHeadTarget2 = !IsHeadTarget2;
			if (IsHeadTarget2)
			{
				Head_Target.SetProp(frame, "Target2", Head_Keys);
				IsHeadTarget2 = false;
			}
		}
		if (Utility.GUIButton("Target3", IsHeadTarget3 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsHeadTarget3 = !IsHeadTarget3;
			if (IsHeadTarget3)
			{
				Head_Target.SetProp(frame, "Target3", Head_Keys);
				IsHeadTarget3 = false;
			}
		}
		if (Utility.GUIButton("Null", IsHeadTargetNull ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsHeadTargetNull = !IsHeadTargetNull;
			if (IsHeadTargetNull)
			{
				Head_Target.SetProp(frame, "TargetNull", Head_Keys);
				IsHeadTargetNull = false;
			}
		}
		EditorGUILayout.EndHorizontal();
		EditorGUI.EndDisabledGroup(); 

		EditorGUILayout.BeginHorizontal();
		if (Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetPreviousKey(frame, Head_Keys));
		}
		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		Rect ctrl = EditorGUILayout.GetControlRect();
		Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		UltiDraw.Begin();

		float startTime = frame.Timestamp - editor.GetWindow() / 2f;
		float endTime = frame.Timestamp + editor.GetWindow() / 2f;
		if (startTime < 0f)
		{
			endTime -= startTime;
			startTime = 0f;
		}
		if (endTime > Data.GetTotalTime())
		{
			startTime -= endTime - Data.GetTotalTime();
			endTime = Data.GetTotalTime();
		}
		startTime = Mathf.Max(0f, startTime);
		endTime = Mathf.Min(Data.GetTotalTime(), endTime);
		int start = Data.GetFrame(startTime).Index;
		int end = Data.GetFrame(endTime).Index;
		int elements = end - start;

		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

		//Keys
		for (int i = 0; i < Head_Keys.Length; i++)
		{
			if (Head_Keys[i])
			{
				top.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				bottom.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.White);
			}
		}

		//Current Pivot
		top.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

		UltiDraw.End();

		EditorGUILayout.EndVertical();
		if (Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetNextKey(frame, Head_Keys));
		}
		EditorGUILayout.EndHorizontal();
		#endregion

		#region RightHand		
		if (Utility.GUIButton("Key (RightHand)", IsKey(frame, RightHand_Keys) ? UltiDraw.Cyan : UltiDraw.DarkGrey, IsKey(frame, RightHand_Keys) ? UltiDraw.Black : UltiDraw.White))
		{
			ToggleKey(RightHand_Target, frame, RightHand_Keys);
		}

		EditorGUI.BeginDisabledGroup(!IsKey(frame, RightHand_Keys));

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("RightHand Target", GUILayout.Width(100f));
		EditorGUILayout.TextField(RightHand_Target.GetProp(frame));
		if (Utility.GUIButton("Target1", IsRightHandTarget1 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsRightHandTarget1 = !IsRightHandTarget1;
			if (IsRightHandTarget1)
			{
				RightHand_Target.SetProp(frame, "Target1", RightHand_Keys);
				IsRightHandTarget1 = false;
			}
		}
		if (Utility.GUIButton("Target2", IsRightHandTarget2 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsRightHandTarget2 = !IsRightHandTarget2;
			if (IsRightHandTarget2)
			{
				RightHand_Target.SetProp(frame, "Target2", RightHand_Keys);
				IsRightHandTarget2 = false;
			}
		}
		if (Utility.GUIButton("Target3", IsRightHandTarget3 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsRightHandTarget3 = !IsRightHandTarget3;
			if (IsRightHandTarget3)
			{
				RightHand_Target.SetProp(frame, "Target3", RightHand_Keys);
				IsRightHandTarget3 = false;
			}
		}
		if (Utility.GUIButton("Null", IsRightHandTargetNull ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsRightHandTargetNull = !IsRightHandTargetNull;
			if (IsRightHandTargetNull)
			{
				RightHand_Target.SetProp(frame, "TargetNull", RightHand_Keys);
				IsRightHandTargetNull = false;
			}
		}
		EditorGUILayout.EndHorizontal();
		EditorGUI.EndDisabledGroup();

		EditorGUILayout.BeginHorizontal();
		if (Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetPreviousKey(frame, RightHand_Keys));
		}
		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		ctrl = EditorGUILayout.GetControlRect();
		rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		UltiDraw.Begin();

		bottom = new Vector3(0f, rect.yMax, 0f);
		top = new Vector3(0f, rect.yMax - rect.height, 0f);

		//Keys
		for (int i = 0; i < RightHand_Keys.Length; i++)
		{
			if (RightHand_Keys[i])
			{
				top.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				bottom.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.White);
			}
		}

		//Current Pivot
		top.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

		UltiDraw.End();

		EditorGUILayout.EndVertical();
		if (Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetNextKey(frame, RightHand_Keys));
		}
		EditorGUILayout.EndHorizontal();
		#endregion

		#region LeftHand		
		if (Utility.GUIButton("Key (LeftHand)", IsKey(frame, LeftHand_Keys) ? UltiDraw.Cyan : UltiDraw.DarkGrey, IsKey(frame, LeftHand_Keys) ? UltiDraw.Black : UltiDraw.White))
		{
			ToggleKey(LeftHand_Target, frame, LeftHand_Keys);
		}

		EditorGUI.BeginDisabledGroup(!IsKey(frame, LeftHand_Keys));

		EditorGUILayout.BeginHorizontal();
		EditorGUILayout.LabelField("LeftHand Target", GUILayout.Width(100f));
		EditorGUILayout.TextField(LeftHand_Target.GetProp(frame));
		if (Utility.GUIButton("Target1", IsLeftHandTarget1 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsLeftHandTarget1 = !IsLeftHandTarget1;
			if (IsLeftHandTarget1)
			{
				LeftHand_Target.SetProp(frame, "Target1", LeftHand_Keys);
				IsLeftHandTarget1 = false;
			}
		}
		if (Utility.GUIButton("Target2", IsLeftHandTarget2 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsLeftHandTarget2 = !IsLeftHandTarget2;
			if (IsLeftHandTarget2)
			{
				LeftHand_Target.SetProp(frame, "Target2", LeftHand_Keys);
				IsLeftHandTarget2 = false;
			}
		}
		if (Utility.GUIButton("Target3", IsLeftHandTarget3 ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsLeftHandTarget3 = !IsLeftHandTarget3;
			if (IsLeftHandTarget3)
			{
				LeftHand_Target.SetProp(frame, "Target3", LeftHand_Keys);
				IsLeftHandTarget3 = false;
			}
		}
		if (Utility.GUIButton("Null", IsLeftHandTargetNull ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
		{
			IsLeftHandTargetNull = !IsLeftHandTargetNull;
			if (IsLeftHandTargetNull)
			{
				LeftHand_Target.SetProp(frame, "TargetNull", LeftHand_Keys);
				IsLeftHandTargetNull = false;
			}
		}
		EditorGUILayout.EndHorizontal();
		EditorGUI.EndDisabledGroup();

		EditorGUILayout.BeginHorizontal();
		if (Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetPreviousKey(frame, LeftHand_Keys));
		}
		EditorGUILayout.BeginVertical(GUILayout.Height(50f));
		ctrl = EditorGUILayout.GetControlRect();
		rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 50f);
		EditorGUI.DrawRect(rect, UltiDraw.Black);

		UltiDraw.Begin();

		bottom = new Vector3(0f, rect.yMax, 0f);
		top = new Vector3(0f, rect.yMax - rect.height, 0f);

		//Keys
		for (int i = 0; i < LeftHand_Keys.Length; i++)
		{
			if (LeftHand_Keys[i])
			{
				top.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				bottom.x = rect.xMin + (float)(i + 1 - start) / elements * rect.width;
				UltiDraw.DrawLine(top, bottom, UltiDraw.White);
			}
		}

		//Current Pivot
		top.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index - start) / elements * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

		UltiDraw.End();

		EditorGUILayout.EndVertical();
		if (Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 25f, 50f))
		{
			editor.LoadFrame(GetNextKey(frame, LeftHand_Keys));
		}
		EditorGUILayout.EndHorizontal();
		#endregion
	}

	[System.Serializable]
	public class TargetFunction
	{
		public TargetModule Module;
		public int[] References;
		public string[] Props;
		public string[] Specifiers;

		public void Fix()
		{
			Specifiers = new string[Module.Data.Frames.Length];
			for (int i = 0; i < Specifiers.Length; i++)
			{
				Specifiers[i] = string.Empty;
			}
		}

		public TargetFunction(TargetModule module)
		{
			Module = module;
			References = new int[Module.Data.Frames.Length];
			Props = new string[Module.Data.Frames.Length];
			Specifiers = new string[Module.Data.Frames.Length];

			for (int i = 0; i < References.Length; i++)
			{
				References[i] = 60;
			}
			for (int i = 0; i < Props.Length; i++)
			{
				Props[i] = string.Empty;
			}
			for (int i = 0; i < Specifiers.Length; i++)
			{
				Specifiers[i] = string.Empty;
			}
		}

		public void SetReference(Frame frame, int value, bool[] keys)
		{
			if (Module.IsKey(frame, keys) && GetReference(frame) != value)
			{
				References[frame.Index - 1] = value;
				Compute(frame, keys);
			}
		}

		public int GetReference(Frame frame)
		{
			return References[frame.Index - 1];
		}

		public void SetProp(Frame frame, string value, bool[] keys)
		{
			if (Module.IsKey(frame, keys) && GetProp(frame) != value)
			{
				Props[frame.Index - 1] = value;
				Compute(frame, keys);
			}
		}

		public string GetProp(Frame frame)
		{
			return Props[frame.Index - 1];
		}

		public void SetSpecifier(Frame frame, string value, bool[] keys)
		{
			if (Module.IsKey(frame, keys) && GetSpecifier(frame) != value)
			{
				Specifiers[frame.Index - 1] = value;
				Compute(frame, keys);
			}
		}

		public string GetSpecifier(Frame frame)
		{
			return Specifiers[frame.Index - 1];
		}

		public void Compute(Frame frame, bool[] keys)
		{
			Frame current = frame;
			Frame previous = Module.GetPreviousKey(current, keys);
			Frame next = Module.GetNextKey(current, keys);

			if (Module.IsKey(current, keys))
			{
				//Current Frame
				References[current.Index - 1] = GetReference(current);
				Props[current.Index - 1] = GetProp(current);
				Specifiers[current.Index - 1] = GetSpecifier(current);
				//Previous Frames
				if (previous != current)
				{
					for (int i = previous.Index; i < current.Index; i++)
					{
						References[i - 1] = GetReference(previous);
						Props[i - 1] = GetProp(previous);
						Specifiers[i - 1] = GetSpecifier(previous);
					}
				}
				//Next Frames
				if (next != current)
				{
					if (next.Index == Module.Data.GetTotalFrames())
					{
						for (int i = current.Index + 1; i <= next.Index; i++)
						{
							References[i - 1] = GetReference(current);
							Props[i - 1] = GetProp(current);
							Specifiers[i - 1] = GetSpecifier(current);
						}
					}
					else
					{
						for (int i = current.Index + 1; i < next.Index; i++)
						{
							References[i - 1] = GetReference(current);
							Props[i - 1] = GetProp(current);
							Specifiers[i - 1] = GetSpecifier(current);
						}
					}
				}
			}
			else
			{
				for (int i = previous.Index; i < next.Index; i++)
				{
					References[i - 1] = GetReference(previous);
					Props[i - 1] = GetProp(previous);
					Specifiers[i - 1] = GetSpecifier(previous);
				}
			}
		}
	}
}
#endif
