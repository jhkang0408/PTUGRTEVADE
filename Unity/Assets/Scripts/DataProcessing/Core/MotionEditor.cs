#if UNITY_EDITOR
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditorInternal;

[ExecuteInEditMode]
public class MotionEditor : MonoBehaviour
{

	public string Folder = string.Empty;
	public bool IsTrain = true;
	public bool IsPoint = true;
	public MotionData[] Files = new MotionData[0];
	public bool Save = true;
	public Transform Root;
	public Transform Head;
	public Transform Gaze;
	public Transform RightHand;
	public Transform RightHand_IndexFingerTip;
	public Transform LeftHand;
	public Transform LeftHand_IndexFingerTip;
	public Transform[] Targets = new Transform[3];

	public Vector3 Head_Rotation_Offset = Vector3.zero;
	public Vector3 RightHand_Rotation_Offset = Vector3.zero;
	public Vector3 LeftHand_Rotation_Offset = Vector3.zero;

	public bool LeftHanded = false;
	public bool Mirroring = false;
	public bool DrawEE = false;
	public bool DrawRay = false;
	public bool DrawScaling = false;
	public float TargetFramerate = 60f;
	public int RandomSeed = 0;

	public bool Callbacks = true;
	public bool Visualise = true;
	public bool Settings = false;

	public bool Playing = false;
	//private float Timescale = 1f;
	public float Timescale = 1f;
	private float Timestamp = 0f;
	private float Zoom = 1f;

	[SerializeField] private MotionData Data = null;

	public static MotionEditor GetInstance()
	{
		return GameObject.FindObjectOfType<MotionEditor>();
	}

	public void Refresh()
	{
		for (int i = 0; i < Files.Length; i++)
		{
			if (Files[i] == null)
			{
				Debug.Log("Removing missing file from editor.");
				ArrayExtensions.RemoveAt(ref Files, i);
				for (int j = 0; j < EditorSceneManager.sceneCount; j++)
				{
					Scene scene = EditorSceneManager.GetSceneAt(j);
					if (scene != EditorSceneManager.GetActiveScene())
					{
						if (!System.Array.Find(Files, x => x != null && x.GetName() == scene.name))
						{
							EditorSceneManager.CloseScene(scene, true);
							break;
						}
					}
				}
				i--;
			}
		}
		if (Data == null && Files.Length > 0)
		{
			LoadData(Files[0]);
		}
	}

	public void SetCallbacks(bool value)
	{
		if (Callbacks != value)
		{
			Callbacks = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetOffset(Vector3 value)
	{
		if (Data.Offset != value)
		{
			Data.Offset = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetScale(float value)
	{
		if (Data.Scale != value)
		{
			Data.Scale = value;
			LoadFrame(Timestamp);
		}
	}

	public void SetTargetFramerate(float value)
	{
		TargetFramerate = Data == null ? 1 : Mathf.Clamp(value, 1, Data.Framerate);
	}

	public void SetRandomSeed(int value)
	{
		RandomSeed = Mathf.Max(value, 0);
	}

	public int GetCurrentSeed()
	{
		if (RandomSeed == 0)
		{
			Frame frame = GetCurrentFrame();
			return frame == null ? 0 : frame.Index;
		}
		else
		{
			return RandomSeed;
		}
	}

	public float GetWindow()
	{
		return Data == null ? 0f : Zoom * Data.GetTotalTime();
	}

	public Vector3Int GetView()
	{
		float startTime = GetCurrentFrame().Timestamp - GetWindow() / 2f;
		float endTime = GetCurrentFrame().Timestamp + GetWindow() / 2f;
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
		int elements = end - start + 1;
		return new Vector3Int(start, end, elements);
	}

	public MotionData GetData()
	{
		return Data;
	}

	public float RoundToTargetTime(float time)
	{
		return Mathf.RoundToInt(time * TargetFramerate) / TargetFramerate;
	}

	public float CeilToTargetTime(float time)
	{
		return Mathf.CeilToInt(time * TargetFramerate) / TargetFramerate;
	}

	public float FloorToTargetTime(float time)
	{
		return Mathf.FloorToInt(time * TargetFramerate) / TargetFramerate;
	}


	public Frame GetCurrentFrame()
	{
		return Data == null ? null : Data.GetFrame(Timestamp);
	}

	public void Import()
	{
		LoadData((MotionData)null);
		string[] assets = AssetDatabase.FindAssets("t:MotionData", new string[1] { Folder });
		Files = new MotionData[assets.Length];
		for (int i = 0; i < assets.Length; i++)
		{
			Files[i] = (MotionData)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(assets[i]), typeof(MotionData));
		}
	}

	public void LoadData(string name)
	{
		if (Data != null && Data.GetName() == name)
		{
			return;
		}
		MotionData data = System.Array.Find(Files, x => x.GetName() == name);
		if (data == null)
		{
			Debug.Log("Data " + name + " could not be found.");
			return;
		}
		LoadData(data);
	}

	public void LoadData(MotionData data)
	{
		if (Data != data)
		{
			if (Data != null)
			{
				if (Save)
				{
					Data.Save();
				}
				Data.Unload();
			}

			Data = data;
			if (Data != null)
			{
				Data.Load();
				LoadFrame(0f);
			}
		}
	}

	public void LoadPreviousData()
	{
		if (Data == null)
		{
			return;
		}
		LoadData(Files[Mathf.Max(System.Array.FindIndex(Files, x => x == Data) - 1, 0)]);
	}

	public void LoadNextData()
	{
		if (Data == null)
		{
			return;
		}
		LoadData(Files[Mathf.Min(System.Array.FindIndex(Files, x => x == Data) + 1, Files.Length - 1)]);
	}

	public void LoadFrame(float timestamp)
	{
		Timestamp = timestamp;
		Scene scene = Data.GetScene();
		Frame frame = GetCurrentFrame();

		UnityEngine.Random.InitState(frame.Index + GetCurrentSeed());

		Matrix4x4 head_matrix = !Mirroring ? frame.Get_Head() : frame.Get_Head().GetMirror(Axis.XPositive);
		Head.position = head_matrix.GetPosition();
		Head.rotation = head_matrix.GetRotation() * Quaternion.Euler(Head_Rotation_Offset);

		// Gaze
		Gaze.position = (!Mirroring ? frame.Get_GazeOrigin() : frame.Get_GazeOrigin().GetMirror(Axis.XPositive));
		Gaze.forward = (!Mirroring ? frame.Get_GazeDirection() : frame.Get_GazeDirection().GetMirror(Axis.XPositive));

		// RightHand
		Matrix4x4 righthand_matrix = !Mirroring ? frame.Get_RightHand() : frame.Get_LeftHand().GetMirror(Axis.XPositive);
		RightHand.position = righthand_matrix.GetPosition();
		RightHand.rotation = righthand_matrix.GetRotation() * Quaternion.Euler(RightHand_Rotation_Offset);
		// RightHand_IndexFingerTip
		RightHand_IndexFingerTip.position = (!Mirroring ? frame.Get_RightHand_IndexFingerTip() : frame.Get_LeftHand_IndexFingerTip().GetMirror(Axis.XPositive));

		// LeftHand
		Matrix4x4 lefthand_matrix = !Mirroring ? frame.Get_LeftHand() : frame.Get_RightHand().GetMirror(Axis.XPositive);
		LeftHand.position = lefthand_matrix.GetPosition();
		LeftHand.rotation = lefthand_matrix.GetRotation() * Quaternion.Euler(LeftHand_Rotation_Offset);
		// LeftHand_IndexFingerTip
		LeftHand_IndexFingerTip.position = (!Mirroring ? frame.Get_LeftHand_IndexFingerTip() : frame.Get_RightHand_IndexFingerTip().GetMirror(Axis.XPositive));

		// Targets
		Targets[0].position = !Mirroring ? frame.Get_Target(0) : frame.Get_Target(0).GetMirror(Axis.XPositive);
		Targets[1].position = !Mirroring ? frame.Get_Target(1) : frame.Get_Target(1).GetMirror(Axis.XPositive);
		Targets[2].position = !Mirroring ? frame.Get_Target(2) : frame.Get_Target(2).GetMirror(Axis.XPositive);
		// Root
		Root.position = Head.position;
		Root.rotation = Quaternion.LookRotation(Vector3.ProjectOnPlane(Head.forward, Vector3.up).normalized, Vector3.up);

		foreach (GameObject instance in scene.GetRootGameObjects())
		{
			foreach (SceneEvent e in instance.GetComponentsInChildren<SceneEvent>(true))
			{
				if (Callbacks)
				{
					e.Callback(this);
				}
				else
				{
					e.Identity(this);
				}
			}
		}
		foreach (Module m in Data.Modules)
		{
			if (Callbacks)
			{
				m.Callback(this);
			}
		}
	}

	public void LoadFrame(int index)
	{
		LoadFrame(Data.GetFrame(index).Timestamp);
	}

	public void LoadFrame(Frame frame)
	{
		LoadFrame(frame.Index);
	}

	public void PlayAnimation()
	{
		if (Playing)
		{
			return;
		}
		Playing = true;
		EditorCoroutines.StartCoroutine(Play(), this);
	}

	public void StopAnimation()
	{
		if (!Playing)
		{
			return;
		}
		Playing = false;
		EditorCoroutines.StopCoroutine(Play(), this);
	}

	private IEnumerator Play()
	{
		System.DateTime previous = Utility.GetTimestamp();
		while (Data != null)
		{
			float delta = Timescale * (float)Utility.GetElapsedTime(previous);
			if (delta > 1f / TargetFramerate)
			{
				previous = Utility.GetTimestamp();
				LoadFrame(Mathf.Repeat(Timestamp + delta, Data.GetTotalTime()));
			}
			yield return new WaitForSeconds(0f);
		}
	}

	public void Draw()
	{
		if (Data != null)
		{
			for (int i = 0; i < Data.Modules.Length; i++)
			{
				Data.Modules[i].Draw(this);
			}
		}
	}

	void OnRenderObject()
	{
		Draw();
		UltiDraw.Begin();

		string head_target_name= "TargetNull";
		string righthand_target_name= "TargetNull";
		string lefthand_target_name = "TargetNull";
		Vector3 head_target = Vector3.zero;
		Vector3 righthand_target = Vector3.zero;
		Vector3 lefthand_target = Vector3.zero;

		if (DrawEE)
		{
			if (Data != null)
			{
				if (GetData().GetModule(Module.ID.Target) != null)
				{
					head_target_name = ((TargetModule)GetData().GetModule(Module.ID.Target)).Head_Target.Props[GetCurrentFrame().Index - 1];
					if (!Mirroring)
					{
						righthand_target_name = ((TargetModule)GetData().GetModule(Module.ID.Target)).RightHand_Target.Props[GetCurrentFrame().Index - 1];
						lefthand_target_name = ((TargetModule)GetData().GetModule(Module.ID.Target)).LeftHand_Target.Props[GetCurrentFrame().Index - 1];
					}
					else
                    {
						lefthand_target_name = ((TargetModule)GetData().GetModule(Module.ID.Target)).RightHand_Target.Props[GetCurrentFrame().Index - 1];
						righthand_target_name = ((TargetModule)GetData().GetModule(Module.ID.Target)).LeftHand_Target.Props[GetCurrentFrame().Index - 1];
					}
				}

				if (head_target_name != "TargetNull")
                {
					head_target = GameObject.Find(head_target_name).transform.position;
					UltiDraw.DrawSphere(head_target + new Vector3(0, IsPoint == true ? 0.025f : 0.0125f, 0), Quaternion.identity, IsPoint == true ? 0.10f : 0.05f, UltiDraw.Red);
					//UltiDraw.DrawTranslateGizmo(head_target + new Vector3(0, IsPoint == true ? 0.025f : 0.0125f, 0), Quaternion.LookRotation(Head.position - head_target), 0.15f);
				}
				if (righthand_target_name != "TargetNull")
				{
					righthand_target = GameObject.Find(righthand_target_name).transform.position;
					UltiDraw.DrawSphere(righthand_target, Quaternion.identity, IsPoint == true ? 0.10f : 0.05f, UltiDraw.Green);
					//UltiDraw.DrawTranslateGizmo(righthand_target, Quaternion.LookRotation(RightHand.position - righthand_target), 0.15f);
				}
				if (lefthand_target_name != "TargetNull")
                {
					lefthand_target = GameObject.Find(lefthand_target_name).transform.position;
					UltiDraw.DrawSphere(lefthand_target - new Vector3(0, IsPoint == true ? 0.025f : 0.0125f, 0), Quaternion.identity, IsPoint == true ? 0.10f : 0.05f, UltiDraw.Blue);
					//UltiDraw.DrawTranslateGizmo(lefthand_target - new Vector3(0, IsPoint == true ? 0.025f : 0.0125f, 0), Quaternion.LookRotation(LeftHand.position - lefthand_target), 0.15f);
				}
			}

			// Head
			UltiDraw.DrawTranslateGizmo(Head.position, Head.rotation, 0.15f);
			//UltiDraw.DrawSphere(Head.position, Head.rotation, 0.15f, new Color(1f, 0f, 0f, 0.5f));
			UltiDraw.DrawSphere(Head.position, Head.rotation, 0.15f, Color.red);
			// RightHand
			UltiDraw.DrawTranslateGizmo(RightHand.position, RightHand.rotation, 0.075f);
			//UltiDraw.DrawSphere(RightHand.position, RightHand.rotation, 0.10f, new Color(0f, 1f, 0f, 0.5f));
			UltiDraw.DrawSphere(RightHand.position, RightHand.rotation, 0.10f, Color.green);
			// RightHand_IndexFingerTip
			//UltiDraw.DrawSphere(RightHand_IndexFingerTip.position, Quaternion.identity, 0.05f, new Color(0f, 1f, 0f, 0.5f));
			UltiDraw.DrawSphere(RightHand_IndexFingerTip.position, Quaternion.identity, 0.05f, Color.green);
			// LeftHand
			UltiDraw.DrawTranslateGizmo(LeftHand.position, LeftHand.rotation, 0.075f);
			//UltiDraw.DrawSphere(LeftHand.position, LeftHand.rotation, 0.10f, new Color(0f, 0f, 1f, 0.5f));
			UltiDraw.DrawSphere(LeftHand.position, LeftHand.rotation, 0.10f, Color.blue);
			// LeftHand_IndexFingerTip
			//UltiDraw.DrawSphere(LeftHand_IndexFingerTip.position, Quaternion.identity, 0.05f, new Color(0f, 0f, 1f, 0.5f));
			UltiDraw.DrawSphere(LeftHand_IndexFingerTip.position, Quaternion.identity, 0.05f, Color.blue);
		}

		if (DrawRay)
		{
			if (head_target_name != "TargetNull")
            {
				UltiDraw.DrawLine(Head.position, head_target, UltiDraw.White);
			}
			if (righthand_target_name != "TargetNull")
			{
				UltiDraw.DrawLine(Head.position, righthand_target, UltiDraw.White);
			}
			if (lefthand_target_name != "TargetNull")
			{
				UltiDraw.DrawLine(Head.position, lefthand_target, UltiDraw.White);
			}
			UltiDraw.DrawLine(Head.position, Head.position + 2 * Gaze.forward, UltiDraw.Red);
			UltiDraw.DrawLine(Head.position, Head.position + 2 * (RightHand_IndexFingerTip.position - Head.position).normalized, UltiDraw.Green);
			UltiDraw.DrawLine(Head.position, Head.position + 2 * (LeftHand_IndexFingerTip.position - Head.position).normalized, UltiDraw.Blue);
		}

		if (DrawScaling)
		{
			float scale = 0.8f;
			// Head			
			UltiDraw.DrawTranslateGizmo(Head.position, Head.rotation, scale * 0.15f);
			UltiDraw.DrawSphere(Head.position, Head.rotation, scale * 0.15f, UltiDraw.Red);
			// RightHand
			UltiDraw.DrawTranslateGizmo(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand.position)), RightHand.rotation, scale * 0.075f);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand.position)), RightHand.rotation, scale * 0.10f, UltiDraw.Green);
			// RightHand_IndexFingerTip
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand_IndexFingerTip.position)), Quaternion.identity, scale * 0.05f, UltiDraw.Green);
			// LeftHand
			UltiDraw.DrawTranslateGizmo(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand.position)), RightHand.rotation, scale * 0.075f);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand.position)), RightHand.rotation, scale * 0.10f, UltiDraw.Blue);
			// LeftHand_IndexFingerTip                
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand_IndexFingerTip.position)), Quaternion.identity, scale * 0.05f, UltiDraw.Blue);
			// Target
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(head_target)) + scale * new Vector3(0, 0.05f, 0), Quaternion.identity, scale * 0.10f, UltiDraw.Red);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(righthand_target)), Quaternion.identity, scale * 0.10f, UltiDraw.Green);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(lefthand_target)) - scale * new Vector3(0, 0.05f, 0), Quaternion.identity, scale * 0.10f, UltiDraw.Blue);
			
			scale = 1.2f;
			// Head			
			UltiDraw.DrawTranslateGizmo(Head.position, Head.rotation, scale * 0.15f);
			UltiDraw.DrawSphere(Head.position, Head.rotation, scale * 0.15f, UltiDraw.Red);
			// RightHand
			UltiDraw.DrawTranslateGizmo(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand.position)), RightHand.rotation, scale * 0.075f);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand.position)), RightHand.rotation, scale * 0.10f, UltiDraw.Green);
			// RightHand_IndexFingerTip                
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(RightHand_IndexFingerTip.position)), Quaternion.identity, scale * 0.05f, UltiDraw.Green);
			// LeftHand
			UltiDraw.DrawTranslateGizmo(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand.position)), RightHand.rotation, scale * 0.075f);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand.position)), RightHand.rotation, scale * 0.10f, UltiDraw.Blue);
			// LeftHand_IndexFingerTip                
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(LeftHand_IndexFingerTip.position)), Quaternion.identity, scale * 0.05f, UltiDraw.Blue);
			// Target
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(head_target)) + scale * new Vector3(0, 0.05f, 0), Quaternion.identity, scale * 0.10f, UltiDraw.Red);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(righthand_target)), Quaternion.identity, scale * 0.10f, UltiDraw.Green);
			UltiDraw.DrawSphere(Head.TransformPoint(scale * Head.InverseTransformPoint(lefthand_target)) - scale * new Vector3(0, 0.05f, 0), Quaternion.identity, scale * 0.10f, UltiDraw.Blue);
		}
		UltiDraw.End();
	}

	void OnDrawGizmos()
	{
		if (!Application.isPlaying)
		{
			OnRenderObject();
		}
	}

	public void DrawPivot(Rect rect)
	{
		UltiDraw.Begin();
		Frame frame = GetCurrentFrame();
		Vector3 view = GetView();
		Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
		Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);
		float pStart = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp - 1f, 0f, Data.GetTotalTime())).Index - view.x) / (view.z - 1);
		float pEnd = (float)(Data.GetFrame(Mathf.Clamp(frame.Timestamp + 1f, 0f, Data.GetTotalTime())).Index - view.x) / (view.z - 1);
		float pLeft = rect.x + pStart * rect.width;
		float pRight = rect.x + pEnd * rect.width;
		Vector3 pA = new Vector3(pLeft, rect.y, 0f);
		Vector3 pB = new Vector3(pRight, rect.y, 0f);
		Vector3 pC = new Vector3(pLeft, rect.y + rect.height, 0f);
		Vector3 pD = new Vector3(pRight, rect.y + rect.height, 0f);
		UltiDraw.DrawTriangle(pA, pC, pB, UltiDraw.White.Transparent(0.1f));
		UltiDraw.DrawTriangle(pB, pC, pD, UltiDraw.White.Transparent(0.1f));
		top.x = rect.xMin + (float)(frame.Index - view.x) / (view.z - 1) * rect.width;
		bottom.x = rect.xMin + (float)(frame.Index - view.x) / (view.z - 1) * rect.width;
		UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);
		UltiDraw.End();
	}

	[CustomEditor(typeof(MotionEditor))]
	public class MotionEditor_Editor : Editor
	{

		public MotionEditor Target;

		private float RepaintRate = 10f;
		private System.DateTime Timestamp;

		public string[] Names = new string[0];
		public string[] EnumNames = new string[0];
		public string NameFilter = "";

		void Awake()
		{
			Target = (MotionEditor)target;
			Target.Refresh();
			ComputeNames();
			Timestamp = Utility.GetTimestamp();
			EditorApplication.update += EditorUpdate;
		}

		void OnDestroy()
		{
			EditorApplication.update -= EditorUpdate;
			if (Target.Data != null)
			{
				Target.Data.Save();
			}
		}

		public void EditorUpdate()
		{
			if (Utility.GetElapsedTime(Timestamp) >= 1f / RepaintRate)
			{
				Repaint();
				Timestamp = Utility.GetTimestamp();
			}
		}

		public override void OnInspectorGUI()
		{
			Undo.RecordObject(Target, Target.name);
			Inspector();
			if (GUI.changed)
			{
				EditorUtility.SetDirty(Target);
			}
		}

		public void ComputeNames()
		{
			List<string> names = new List<string>();
			List<string> enumNames = new List<string>();
			for (int i = 0; i < Target.Files.Length; i++)
			{
				if (Target.Files[i].GetName().ToLowerInvariant().Contains(NameFilter.ToLowerInvariant()))
				{
					names.Add(Target.Files[i].GetName());
					enumNames.Add("[" + (i + 1) + "]" + " " + Target.Files[i].GetName());
				}
			}
			Names = names.ToArray();
			EnumNames = enumNames.ToArray();
		}

		public void SetNameFilter(string filter)
		{
			if (NameFilter != filter)
			{
				NameFilter = filter;
				ComputeNames();
			}
		}

		public void Import()
		{
			Target.Import();
			ComputeNames();
		}

		public void Inspector()
		{
			Target.Refresh();

			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using (new EditorGUILayout.VerticalScope("Box"))
			{
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();

					EditorGUILayout.BeginHorizontal();
					Target.Folder = EditorGUILayout.TextField("Folder", "Assets/" + Target.Folder.Substring(Mathf.Min(7, Target.Folder.Length)));
					if (Utility.GUIButton("Import", UltiDraw.DarkGrey, UltiDraw.White))
					{
						Import();
					}
					EditorGUILayout.EndHorizontal();
					{
						string Folder_temp = Target.Folder.Split('/')[Target.Folder.Split('/').Length - 1];
						string[] folderParts = Folder_temp.Split(' ');

						switch (folderParts[0])
						{
							case "Pointing":
								Target.IsPoint = true;
								break;
							case "Touching":
								Target.IsPoint = false;
								break;
							case "Transition":
								Target.IsPoint = folderParts.Length > 1 && folderParts[1] == "(Pointing)";
								break;
						}
						Target.IsPoint = (bool)EditorGUILayout.Toggle("Pointing (True) or Touching (False)", Target.IsPoint);
					}
					{
						if (Target.Folder.Split('/')[2] == "Train")
						{
							Target.IsTrain = true;
						}
						else
						{
							Target.IsTrain = false;
						}
						Target.IsTrain = (bool)EditorGUILayout.Toggle("Training Subject", Target.IsTrain);
					}
					Target.Root = (Transform)EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true);
					Target.Head = (Transform)EditorGUILayout.ObjectField("Head", Target.Head, typeof(Transform), true);
					Target.Gaze = (Transform)EditorGUILayout.ObjectField("Gaze", Target.Gaze, typeof(Transform), true);
					Target.RightHand = (Transform)EditorGUILayout.ObjectField("RightHand", Target.RightHand, typeof(Transform), true);
					Target.RightHand_IndexFingerTip = (Transform)EditorGUILayout.ObjectField("RightHand_IndexFingerTip", Target.RightHand_IndexFingerTip, typeof(Transform), true);
					Target.LeftHand = (Transform)EditorGUILayout.ObjectField("LeftHand", Target.LeftHand, typeof(Transform), true);
					Target.LeftHand_IndexFingerTip = (Transform)EditorGUILayout.ObjectField("LeftHand_IndexFingerTip", Target.LeftHand_IndexFingerTip, typeof(Transform), true);
					for (int i = 0; i < Target.Targets.Length; i++)
					{
						Target.Targets[i] = (Transform)EditorGUILayout.ObjectField("Target" + (i + 1).ToString(), Target.Targets[i], typeof(Transform), true);
						Target.Targets[i].localScale = (Target.IsPoint == true ? 0.10f : 0.05f) * Vector3.one;
					}
					Target.Head_Rotation_Offset = (Vector3)EditorGUILayout.Vector3Field("Head Rotation Offset", Target.Head_Rotation_Offset);
					Target.RightHand_Rotation_Offset = (Vector3)EditorGUILayout.Vector3Field("RightHand Rotation Offset", Target.RightHand_Rotation_Offset);
					Target.LeftHand_Rotation_Offset = (Vector3)EditorGUILayout.Vector3Field("LeftHand Rotation Offset", Target.LeftHand_Rotation_Offset);

					EditorGUILayout.BeginHorizontal();
					if (Utility.GUIButton("Left-Handed", Target.LeftHanded ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.LeftHanded = !Target.LeftHanded;
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					if (Utility.GUIButton("Mirroring", Target.Mirroring ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.Mirroring = !Target.Mirroring;
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					if (Utility.GUIButton("Draw EE", Target.DrawEE ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.DrawEE = !Target.DrawEE;
					}
					if (Utility.GUIButton("Draw Ray", Target.DrawRay ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.DrawRay = !Target.DrawRay;
					}
					if (Utility.GUIButton("Draw Scaling", Target.DrawScaling ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.DrawScaling = !Target.DrawScaling;
					}
					EditorGUILayout.EndHorizontal();

					SetNameFilter(EditorGUILayout.TextField("Name Filter", NameFilter));

					if (Target.Data != null)
					{
						Frame frame = Target.GetCurrentFrame();

						Utility.SetGUIColor(UltiDraw.Grey);
						using (new EditorGUILayout.VerticalScope("Box"))
						{
							Utility.ResetGUIColor();

							EditorGUILayout.BeginHorizontal();
							GUILayout.FlexibleSpace();
							int selectIndex = EditorGUILayout.Popup(System.Array.FindIndex(Names, x => x == Target.Data.GetName()), EnumNames);
							if (selectIndex != -1)
							{
								Target.LoadData(Names[selectIndex]);
							}
							if (Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.LoadPreviousData();
								NameFilter = string.Empty;
							}
							if (Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.LoadNextData();
							}
							int sliderIndex = EditorGUILayout.IntSlider(System.Array.FindIndex(Target.Files, x => x == Target.Data) + 1, 1, Target.Files.Length);
							if (Event.current.type == EventType.Used)
							{
								Target.LoadData(Target.Files[sliderIndex - 1]);
							}
							EditorGUILayout.LabelField("/ " + Target.Files.Length, GUILayout.Width(60f));
							if (Utility.GUIButton("Save", Target.Save ? UltiDraw.Magenta : UltiDraw.LightGrey, UltiDraw.Black))
							{
								Target.Save = !Target.Save;
							}
							GUILayout.FlexibleSpace();
							EditorGUILayout.EndHorizontal();

							Utility.SetGUIColor(UltiDraw.Mustard);
							using (new EditorGUILayout.VerticalScope("Box"))
							{
								Utility.ResetGUIColor();

								EditorGUILayout.BeginHorizontal();
								EditorGUILayout.LabelField("Data", GUILayout.Width(50f));
								EditorGUILayout.ObjectField(Target.Data, typeof(MotionData), true);
								if (Utility.GUIButton("Export", Target.Data.Export ? UltiDraw.Cyan : UltiDraw.Grey, Target.Data.Export ? UltiDraw.Black : UltiDraw.LightGrey))
								{
									Target.Data.Export = !Target.Data.Export;
								}
								EditorGUILayout.EndHorizontal();

								EditorGUILayout.BeginHorizontal();
								GUILayout.FlexibleSpace();
								EditorGUILayout.LabelField("Frames: " + Target.Data.GetTotalFrames(), GUILayout.Width(100f));
								EditorGUILayout.LabelField("Time: " + Target.Data.GetTotalTime().ToString("F3") + "s", GUILayout.Width(100f));
								EditorGUILayout.LabelField("Framerate: " + Target.Data.Framerate.ToString("F1") + "Hz", GUILayout.Width(130f));
								EditorGUILayout.LabelField("Target Framerate: ", GUILayout.Width(100f));
								Target.SetTargetFramerate(EditorGUILayout.FloatField(Target.TargetFramerate, GUILayout.Width(40f)));
								EditorGUILayout.LabelField("Random Seed: ", GUILayout.Width(80f));
								Target.SetRandomSeed(EditorGUILayout.IntField(Target.RandomSeed, GUILayout.Width(40f)));
								GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();
							}

							if (Utility.GUIButton("Add Sequence", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.Data.AddSequence();
							}
							for (int i = 0; i < Target.Data.Sequences.Length; i++)
							{
								Utility.SetGUIColor(UltiDraw.White);
								using (new EditorGUILayout.VerticalScope("Box"))
								{
									Utility.ResetGUIColor();

									EditorGUILayout.BeginHorizontal();
									GUILayout.FlexibleSpace();
									if (Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f))
									{
										Target.Data.Sequences[i].SetStart(frame.Index);
									}
									EditorGUILayout.LabelField("Start", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetStart(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].Start, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
									EditorGUILayout.LabelField("End", GUILayout.Width(50f));
									Target.Data.Sequences[i].SetEnd(Mathf.Clamp(EditorGUILayout.IntField(Target.Data.Sequences[i].End, GUILayout.Width(100f)), 1, Target.Data.GetTotalFrames()));
									if (Utility.GUIButton("X", Color.cyan, Color.black, 20f, 15f))
									{
										Target.Data.Sequences[i].SetEnd(frame.Index);
									}

									if (Utility.GUIButton("X", UltiDraw.DarkRed, Color.black, 40f, 15f))
									{
										Target.Data.RemoveSequence(Target.Data.Sequences[i]);
										i--;
									}
									GUILayout.FlexibleSpace();
									EditorGUILayout.EndHorizontal();
								}
							}

							EditorGUILayout.BeginVertical(GUILayout.Height(25f));
							Rect ctrl = EditorGUILayout.GetControlRect();
							Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 25f);
							EditorGUI.DrawRect(rect, UltiDraw.Black);

							//View
							Vector3 view = Target.GetView();

							//Sequences
							UltiDraw.Begin();
							for (int i = 0; i < Target.Data.Sequences.Length; i++)
							{
								float _start = (float)(Mathf.Clamp(Target.Data.Sequences[i].Start, view.x, view.y) - view.x) / view.z;
								float _end = (float)(Mathf.Clamp(Target.Data.Sequences[i].End, view.x, view.y) - view.x) / view.z;
								float left = rect.x + _start * rect.width;
								float right = rect.x + _end * rect.width;
								Vector3 a = new Vector3(left, rect.y, 0f);
								Vector3 b = new Vector3(right, rect.y, 0f);
								Vector3 c = new Vector3(left, rect.y + rect.height, 0f);
								Vector3 d = new Vector3(right, rect.y + rect.height, 0f);
								UltiDraw.DrawTriangle(a, c, b, UltiDraw.Yellow.Transparent(0.25f));
								UltiDraw.DrawTriangle(b, c, d, UltiDraw.Yellow.Transparent(0.25f));
							}
							UltiDraw.End();

							//Current Pivot
							Target.DrawPivot(rect);

							EditorGUILayout.EndVertical();

							Utility.SetGUIColor(UltiDraw.DarkGrey);
							using (new EditorGUILayout.VerticalScope("Box"))
							{
								Utility.ResetGUIColor();
								EditorGUILayout.BeginHorizontal();
								//GUILayout.FlexibleSpace();
								if (Target.Playing)
								{
									if (Utility.GUIButton("||", Color.red, Color.black, 50f, 20f))
									{
										Target.StopAnimation();
									}
								}
								else
								{
									if (Utility.GUIButton("|>", Color.green, Color.black, 50f, 20f))
									{
										Target.PlayAnimation();
									}
								}
								if (Utility.GUIButton("<<", UltiDraw.Grey, UltiDraw.White, 30f, 20f))
								{
									Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f, 0f));
								}
								if (Utility.GUIButton("<", UltiDraw.Grey, UltiDraw.White, 20f, 20f))
								{
									Target.LoadFrame(Mathf.Max(frame.Timestamp - 1f / Target.TargetFramerate, 0f));
								}
								if (Utility.GUIButton(">", UltiDraw.Grey, UltiDraw.White, 20f, 20f))
								{
									Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f / Target.TargetFramerate, Target.Data.GetTotalTime()));
									Debug.Log("frame.Timestamp=" + frame.Timestamp);
									Debug.Log("1f/Target.TargetFramerate=" + 1f / Target.TargetFramerate);
									Debug.Log("frame.Timestamp + 1f/Target.TargetFramerate=" + (frame.Timestamp + 1f / Target.TargetFramerate));
								}
								if (Utility.GUIButton(">>", UltiDraw.Grey, UltiDraw.White, 30f, 20f))
								{
									Target.LoadFrame(Mathf.Min(frame.Timestamp + 1f, Target.Data.GetTotalTime()));
								}
								int index = EditorGUILayout.IntSlider(frame.Index, 1, Target.Data.GetTotalFrames());
								if (index != frame.Index)
								{
									Target.LoadFrame(index);
								}
								EditorGUILayout.LabelField(frame.Timestamp.ToString("F3") + "s", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
								//GUILayout.FlexibleSpace();
								EditorGUILayout.EndHorizontal();

								EditorGUILayout.BeginHorizontal();
								EditorGUILayout.LabelField("Timescale", Utility.GetFontColor(Color.white), GUILayout.Width(60f), GUILayout.Height(20f));
								Target.Timescale = EditorGUILayout.FloatField(Target.Timescale, GUILayout.Width(45f), GUILayout.Height(16f));
								EditorGUILayout.LabelField("Zoom", Utility.GetFontColor(Color.white), GUILayout.Width(52f));
								Target.Zoom = EditorGUILayout.Slider(Target.Zoom, 0f, 1f);
								EditorGUILayout.LabelField((100 - Mathf.RoundToInt(100f * (1f - Target.Zoom))) + "%", Utility.GetFontColor(Color.white), GUILayout.Width(50f));
								EditorGUILayout.EndHorizontal();
							}

							EditorGUILayout.BeginHorizontal();
							if (Utility.GUIButton("Visualise", Target.Visualise ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
							{
								Target.Visualise = !Target.Visualise;
							}
							if (Utility.GUIButton("Callbacks", Target.Callbacks ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black))
							{
								Target.SetCallbacks(!Target.Callbacks);
							}
							EditorGUILayout.EndHorizontal();

							EditorGUILayout.BeginHorizontal();
							if (Utility.GUIButton("Inspect All", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.Data.InspectAll(true);
							}
							if (Utility.GUIButton("Inspect None", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.Data.InspectAll(false);
							}
							if (Utility.GUIButton("Visualise All", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.Data.VisualiseAll(true);
							}
							if (Utility.GUIButton("Visualise None", UltiDraw.DarkGrey, UltiDraw.White))
							{
								Target.Data.VisualiseAll(false);
							}
							EditorGUILayout.EndHorizontal();

							for (int i = 0; i < Target.Data.Modules.Length; i++)
							{
								Target.Data.Modules[i].Inspector(Target);
							}

							Utility.ResetGUIColor();
							Utility.SetGUIColor(UltiDraw.Cyan);
							int module = EditorGUILayout.Popup(0, ArrayExtensions.Concat(new string[1] { "Add Module..." }, Module.GetIDNames()));
							if (module > 0)
							{
								Target.Data.AddModule(((Module.ID)(module - 1)));
							}
							Utility.ResetGUIColor();
						}
					}
				}
			}
		}
	}
}
#endif