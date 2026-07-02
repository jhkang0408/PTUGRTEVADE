#if UNITY_EDITOR
using System.Threading;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEditor;
using System;

public abstract class MotionExporter : EditorWindow
{
	public static EditorWindow Window;
	public static Vector2 Scroll;

	public float Framerate = 30f;
	public int BatchSize = 16;

	public enum Split { Train, Test }
	public Split ExportSplit = Split.Train;

	public bool Mirrored = false;

	private string[] Subjects = new string[0];
	private int SubjectIndex = 0;

	private class Category
	{
		public string Name;
		public string Path;
		public bool Enabled = true;
		public bool Foldout = false;
		public List<MotionData> Files = new List<MotionData>();
		public List<bool> Selected = new List<bool>();
	}
	private List<Category> Categories = new List<Category>();

	public MotionEditor Editor = null;

	private bool Initialized = false;
	private string StatusLabel = string.Empty;
	private float Progress = 0f;
	private float Performance = 0f;

	protected bool IsTrain = true;

	private static bool Exporting = false;
	private static string Separator = " ";
	private static string Accuracy = "F5";

	private const string TrainRoot = "Assets/Data/Train";
	private const string TestRoot = "Assets/Data/Test";


	public void OnInspectorUpdate()
	{
		Repaint();
	}

	void OnGUI()
	{
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Editor = GameObject.FindObjectOfType<MotionEditor>();

		if (Editor == null)
		{
			Utility.SetGUIColor(UltiDraw.Black);
			using (new EditorGUILayout.VerticalScope("Box"))
			{
				Utility.ResetGUIColor();
				Utility.SetGUIColor(UltiDraw.Grey);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();
					Utility.SetGUIColor(UltiDraw.Orange);
					using (new EditorGUILayout.VerticalScope("Box"))
					{
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Exporter");
					}
					EditorGUILayout.LabelField("No Motion Editor found in scene.");
				}
			}
			EditorGUILayout.EndScrollView();
			return;
		}

		if (!Initialized)
		{
			Load();
		}

		Utility.SetGUIColor(UltiDraw.Black);
		using (new EditorGUILayout.VerticalScope("Box"))
		{
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using (new EditorGUILayout.VerticalScope("Box"))
			{
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Exporter");
				}

				Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
				BatchSize = Mathf.Max(1, EditorGUILayout.IntField("Batch Size", BatchSize));

				Split newSplit = (Split)EditorGUILayout.EnumPopup("Split", ExportSplit);
				if (newSplit != ExportSplit)
				{
					ExportSplit = newSplit;
					SubjectIndex = 0;
					Load();
				}

				if (ExportSplit == Split.Test)
				{
					if (Subjects.Length == 0)
					{
						EditorGUILayout.LabelField("No test subjects found under " + TestRoot + ".");
					}
					else
					{
						int newSubject = EditorGUILayout.Popup("Subject", Mathf.Clamp(SubjectIndex, 0, Subjects.Length - 1), Subjects);
						if (newSubject != SubjectIndex)
						{
							SubjectIndex = newSubject;
							Load();
						}
					}
				}
				else
				{
					Mirrored = EditorGUILayout.Toggle("Mirrored", Mirrored);
				}

				Utility.SetGUIColor(UltiDraw.White);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Export Path: " + GetOutputRoot());
				}

				Utility.SetGUIColor(UltiDraw.LightGrey);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();
					Utility.SetGUIColor(UltiDraw.Cyan);
					using (new EditorGUILayout.VerticalScope("Box"))
					{
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField("Categories [" + Categories.Count + "]");
					}

					if (Categories.Count == 0)
					{
						EditorGUILayout.LabelField("No categories found. Press Reload.");
					}
					else
					{
						for (int c = 0; c < Categories.Count; c++)
						{
							Category cat = Categories[c];
							Utility.SetGUIColor(cat.Enabled ? UltiDraw.Gold : UltiDraw.White);
							using (new EditorGUILayout.VerticalScope("Box"))
							{
								Utility.ResetGUIColor();
								EditorGUILayout.BeginHorizontal();
								cat.Enabled = EditorGUILayout.Toggle(cat.Enabled, GUILayout.Width(16f));
								cat.Foldout = EditorGUILayout.Foldout(cat.Foldout, cat.Name + "  [" + CountSelected(cat) + "/" + cat.Files.Count + "]");
								EditorGUILayout.EndHorizontal();

								if (cat.Foldout)
								{
									EditorGUILayout.BeginHorizontal();
									if (Utility.GUIButton("All", UltiDraw.DarkGrey, UltiDraw.White))
									{
										for (int i = 0; i < cat.Selected.Count; i++) cat.Selected[i] = true;
									}
									if (Utility.GUIButton("None", UltiDraw.DarkGrey, UltiDraw.White))
									{
										for (int i = 0; i < cat.Selected.Count; i++) cat.Selected[i] = false;
									}
									EditorGUILayout.EndHorizontal();

									for (int i = 0; i < cat.Files.Count; i++)
									{
										EditorGUILayout.BeginHorizontal();
										EditorGUILayout.LabelField((i + 1) + " - " + cat.Files[i].GetName(), GUILayout.Width(220f));
										GUILayout.FlexibleSpace();
										if (Utility.GUIButton("O", cat.Selected[i] ? UltiDraw.DarkGreen : UltiDraw.DarkRed, UltiDraw.White, 50f))
										{
											cat.Selected[i] = !cat.Selected[i];
										}
										EditorGUILayout.EndHorizontal();
									}
								}
							}
						}
					}
				}

				if (!Exporting)
				{
					if (Utility.GUIButton("Reload", UltiDraw.DarkGrey, UltiDraw.White))
					{
						Load();
					}
					if (Utility.GUIButton("Export Data", UltiDraw.DarkGrey, UltiDraw.White))
					{
						this.StartCoroutine(ExportData());
					}
				}
				else
				{
					EditorGUILayout.LabelField("Exporting: " + StatusLabel);

					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, Progress * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Transparent(0.75f));

					EditorGUILayout.LabelField("Frames Per Second: " + Performance.ToString("F3"));

					if (Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White))
					{
						Exporting = false;
					}
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}


	public void Load()
	{
		Initialized = true;
		Categories = new List<Category>();

		if (ExportSplit == Split.Test)
		{
			RefreshSubjects();
		}

		string root;
		if (ExportSplit == Split.Train)
		{
			root = TrainRoot;
		}
		else
		{
			if (Subjects.Length == 0)
			{
				return;
			}
			root = TestRoot + "/" + Subjects[Mathf.Clamp(SubjectIndex, 0, Subjects.Length - 1)];
		}

		if (!AssetDatabase.IsValidFolder(root))
		{
			return;
		}

		string[] catFolders = AssetDatabase.GetSubFolders(root);
		Array.Sort(catFolders, StringComparer.Ordinal);
		foreach (string catPath in catFolders)
		{
			Category cat = new Category();
			cat.Path = catPath;
			cat.Name = catPath.Substring(catPath.LastIndexOf('/') + 1);

			string[] guids = AssetDatabase.FindAssets("t:MotionData", new string[] { catPath });
			List<MotionData> data = new List<MotionData>();
			foreach (string g in guids)
			{
				string p = AssetDatabase.GUIDToAssetPath(g);
				MotionData md = (MotionData)AssetDatabase.LoadAssetAtPath(p, typeof(MotionData));
				if (md != null)
				{
					data.Add(md);
				}
			}
			data.Sort((a, b) => NaturalCompare(a.GetName(), b.GetName()));
			foreach (MotionData md in data)
			{
				cat.Files.Add(md);
				cat.Selected.Add(md.Export);
			}
			cat.Enabled = cat.Files.Count > 0;
			Categories.Add(cat);
		}
	}

	private void RefreshSubjects()
	{
		List<string> names = new List<string>();
		if (AssetDatabase.IsValidFolder(TestRoot))
		{
			string[] subs = AssetDatabase.GetSubFolders(TestRoot);
			Array.Sort(subs, StringComparer.Ordinal);
			foreach (string s in subs)
			{
				names.Add(s.Substring(s.LastIndexOf('/') + 1));
			}
		}
		Subjects = names.ToArray();
		if (SubjectIndex >= Subjects.Length)
		{
			SubjectIndex = 0;
		}
	}

	private string GetNetworkRoot()
	{
		return GetExportPath() + "/" + NetworkName;
	}

	private string GetOutputRoot()
	{
		if (ExportSplit == Split.Train)
		{
			return GetNetworkRoot() + "/Train";
		}
		string subject = (Subjects.Length > 0 && SubjectIndex < Subjects.Length) ? Subjects[SubjectIndex] : "<subject>";
		return GetNetworkRoot() + "/Test/" + subject;
	}

	private static int CountSelected(Category cat)
	{
		int n = 0;
		for (int i = 0; i < cat.Selected.Count; i++)
		{
			if (cat.Selected[i])
			{
				n += 1;
			}
		}
		return n;
	}

	private static int NaturalCompare(string a, string b)
	{
		int ai, bi;
		bool an = int.TryParse(a, out ai);
		bool bn = int.TryParse(b, out bi);
		if (an && bn)
		{
			return ai.CompareTo(bi);
		}
		return string.CompareOrdinal(a, b);
	}

	private static bool DeriveIsPoint(string categoryName)
	{
		string n = categoryName.ToLowerInvariant();
		if (n.Contains("pointing"))
		{
			return true;
		}
		if (n.Contains("touching"))
		{
			return false;
		}
		return true;
	}


	private StreamWriter CreateFile(string folder, string name)
	{
		if (!System.IO.Directory.Exists(folder))
		{
			System.IO.Directory.CreateDirectory(folder);
		}
		return File.CreateText(folder + "/" + name + ".txt");
	}


	public class Data
	{
		public StreamWriter File, Norm, Labels;

		public RunningStatistics[] Statistics = null;

		private Queue<float[]> Buffer = new Queue<float[]>();
		private Task Writer = null;

		private float[] Values = new float[0];
		private string[] Names = new string[0];
		private float[] Weights = new float[0];
		private int Dim = 0;

		private bool Finished = false;
		private bool Setup = false;

		public Data(StreamWriter file, StreamWriter norm, StreamWriter labels)
		{
			File = file;
			Norm = norm;
			Labels = labels;
			Writer = Task.Factory.StartNew(() => WriteData());
		}

		public void Feed(float value, string name, float weight = 1f)
		{
			if (!Setup)
			{
				ArrayExtensions.Add(ref Values, value);
				ArrayExtensions.Add(ref Names, name);
				ArrayExtensions.Add(ref Weights, weight);
			}
			else
			{
				Dim += 1;
				Values[Dim - 1] = value;
			}
		}

		public void Feed(float[] values, string name, float weight = 1f)
		{
			for (int i = 0; i < values.Length; i++)
			{
				Feed(values[i], name + (i + 1), weight);
			}
		}

		public void Feed(bool[] values, string name, float weight = 1f)
		{
			for (int i = 0; i < values.Length; i++)
			{
				Feed(values[i] ? 1f : 0f, name + (i + 1), weight);
			}
		}

		public void Feed(float[,] values, string name, float weight = 1f)
		{
			for (int i = 0; i < values.GetLength(0); i++)
			{
				for (int j = 0; j < values.GetLength(1); j++)
				{
					Feed(values[i, j], name + (i * values.GetLength(1) + j + 1), weight);
				}
			}
		}

		public void Feed(bool[,] values, string name, float weight = 1f)
		{
			for (int i = 0; i < values.GetLength(0); i++)
			{
				for (int j = 0; j < values.GetLength(1); j++)
				{
					Feed(values[i, j] ? 1f : 0f, name + (i * values.GetLength(1) + j + 1), weight);
				}
			}
		}

		public void Feed(Vector2 value, string name, float weight = 1f)
		{
			Feed(value.x, name + "X", weight);
			Feed(value.y, name + "Y", weight);
		}

		public void Feed(Vector3 value, string name, float weight = 1f)
		{
			Feed(value.x, name + "X", weight);
			Feed(value.y, name + "Y", weight);
			Feed(value.z, name + "Z", weight);
		}

		public void FeedXY(Vector3 value, string name, float weight = 1f)
		{
			Feed(value.x, name + "X", weight);
			Feed(value.y, name + "Y", weight);
		}

		public void FeedXZ(Vector3 value, string name, float weight = 1f)
		{
			Feed(value.x, name + "X", weight);
			Feed(value.z, name + "Z", weight);
		}

		public void FeedYZ(Vector3 value, string name, float weight = 1f)
		{
			Feed(value.y, name + "Y", weight);
			Feed(value.z, name + "Z", weight);
		}

		private void WriteData()
		{
			while (Exporting && (!Finished || Buffer.Count > 0))
			{
				if (Buffer.Count > 0)
				{
					float[] item;
					lock (Buffer)
					{
						item = Buffer.Dequeue();
					}
					//Update Mean and Std
					for (int i = 0; i < item.Length; i++)
					{
						Statistics[i].Add(item[i]);
					}
					//Write to File
					File.WriteLine(String.Join(Separator, Array.ConvertAll(item, x => x.ToString(Accuracy))));
				}
				else
				{
					Thread.Sleep(1);
				}
			}
		}

		public void Store()
		{
			if (!Setup)
			{
				//Setup Mean and Std
				Statistics = new RunningStatistics[Values.Length];
				for (int i = 0; i < Statistics.Length; i++)
				{
					Statistics[i] = new RunningStatistics();
				}

				//Write Labels
				for (int i = 0; i < Names.Length; i++)
				{
					Labels.WriteLine("[" + i + "]" + " " + Names[i]);
				}
				Labels.Close();

				Setup = true;
			}

			//Enqueue Sample
			float[] item = (float[])Values.Clone();
			lock (Buffer)
			{
				Buffer.Enqueue(item);
			}

			//Reset Running Index
			Dim = 0;
		}

		public void Finish()
		{
			Finished = true;

			Task.WaitAll(Writer);

			File.Close();

			if (Setup)
			{
				//Write Mean
				float[] mean = new float[Statistics.Length];
				for (int i = 0; i < mean.Length; i++)
				{
					mean[i] = Statistics[i].Mean();
				}
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(mean, x => x.ToString(Accuracy))));

				//Write Std
				float[] std = new float[Statistics.Length];
				for (int i = 0; i < std.Length; i++)
				{
					std[i] = Statistics[i].Std();
				}
				Norm.WriteLine(String.Join(Separator, Array.ConvertAll(std, x => x.ToString(Accuracy))));
			}

			Norm.Close();
		}
	}

	[Serializable]
	public class LabelGroup
	{

		public string[] Labels;

		private int[] Indices;

		public LabelGroup(params string[] labels)
		{
			Labels = labels;
		}

		public string GetID()
		{
			string id = string.Empty;
			for (int i = 0; i < Labels.Length; i++)
			{
				id += Labels[i];
			}
			return id;
		}

		public void Setup(string[] references)
		{
			List<int> indices = new List<int>();
			for (int i = 0; i < references.Length; i++)
			{
				if (ArrayExtensions.Contains(ref Labels, references[i]))
				{
					indices.Add(i);
				}
			}
			Indices = indices.ToArray();
		}

		public float Filter(float[] values)
		{
			float value = 0f;
			for (int i = 0; i < Indices.Length; i++)
			{
				value += values[Indices[i]];
			}
			if (value > 1f)
			{
				Debug.Log("Value larger than expected.");
			}
			return value;
		}
	}
	private string GetExportPath()
	{
		string path = Application.dataPath;
		path = path.Substring(0, path.LastIndexOf("/"));
		path += "/Export";
		return path;
	}


	protected abstract void WriteFrame(Data X, Sample current, Sample initial);

	protected abstract string NetworkName { get; }



	private IEnumerator ExportData()
	{
		if (Editor == null)
		{
			Debug.Log("No editor found.");
			yield break;
		}

		string exportRoot = GetExportPath();
		if (!System.IO.Directory.Exists(exportRoot))
		{
			Debug.Log("No export folder found at " + exportRoot + ".");
			yield break;
		}

		Exporting = true;
		Progress = 0f;

		// Wipe only this network's folder before exporting (other networks are left intact).
		string networkRoot = GetNetworkRoot();
		if (System.IO.Directory.Exists(networkRoot))
		{
			System.IO.Directory.Delete(networkRoot, true);
		}
		Debug.Log("Cleared network export folder: " + networkRoot);

		string outputRoot = GetOutputRoot();
		System.IO.Directory.CreateDirectory(outputRoot);

		IsTrain = (ExportSplit == Split.Train);

		int total = 0;
		int items = 0;
		DateTime timestamp = Utility.GetTimestamp();

		MotionData editorData = Editor.GetData();
		bool editorSave = Editor.Save;
		float editorRate = Editor.TargetFramerate;
		int editorSeed = Editor.RandomSeed;
		bool editorIsPoint = Editor.IsPoint;
		bool editorIsTrain = Editor.IsTrain;
		bool editorMirroring = Editor.Mirroring;
		string editorFolder = Editor.Folder;
		Editor.Save = false;
		Editor.SetTargetFramerate(Framerate);

		for (int c = 0; c < Categories.Count; c++)
		{
			if (!Exporting)
			{
				break;
			}
			Category cat = Categories[c];
			if (!cat.Enabled || CountSelected(cat) == 0)
			{
				continue;
			}

			Editor.IsTrain = IsTrain;
			// Point the editor at this category so its live visualization (target sphere
			// size, derived from the folder name) matches the category being exported.
			Editor.Folder = cat.Path;

			int variantCount = (IsTrain && Mirrored) ? 2 : 1;
			for (int v = 0; v < variantCount; v++)
			{
				if (!Exporting)
				{
					break;
				}
				bool mirror = (v == 1);
				Editor.Mirroring = mirror;

				string catFolder = outputRoot + "/" + cat.Name + (mirror ? " (mirrored)" : "");
				Data X = new Data(CreateFile(catFolder, "Data"), CreateFile(catFolder, "DataNorm"), CreateFile(catFolder, "DataLabels"));
				StreamWriter S = CreateFile(catFolder, "Sequences");
				int sequence = 0;

				for (int f = 0; f < cat.Files.Count; f++)
				{
					if (!Exporting)
					{
						break;
					}
					if (!cat.Selected[f])
					{
						continue;
					}

					MotionData file = cat.Files[f];
					StatusLabel = cat.Name + (mirror ? " (mirrored)" : "") + " / " + file.GetName();
					Editor.LoadData(file);
					while (!Editor.GetData().GetScene().isLoaded)
					{
						yield return new WaitForSeconds(0f);
					}

					if (!Exporting)
					{
						break;
					}

					sequence += 1;
					Sequence seq = Editor.GetData().Sequences[0];

					Sample initial = new Sample(Editor, Editor.CeilToTargetTime(Editor.GetData().GetFrame(seq.Start).Timestamp) + 1 / Framerate);
					float start = Editor.RoundToTargetTime(Editor.GetData().GetFrame(seq.Start).Timestamp);
					float end = Editor.RoundToTargetTime(Editor.GetData().GetFrame(seq.End).Timestamp);
					int sample = 1;

					while (start + sample / Framerate <= end)
					{
						if (!Exporting)
						{
							break;
						}

						// Set per frame so a re-derive during a yield cannot corrupt it.
						Editor.Mirroring = mirror;
						Editor.IsPoint = DeriveIsPoint(cat.Name);
						Sample current = new Sample(Editor, start + sample / Framerate);
						S.WriteLine(sequence.ToString());
						WriteFrame(X, current, initial);

						sample += 1;
						Progress = (sample / Framerate) / (end - start);
						total += 1;
						items += 1;
						if (items >= BatchSize)
						{
							Performance = items / (float)Utility.GetElapsedTime(timestamp);
							timestamp = Utility.GetTimestamp();
							items = 0;
							yield return new WaitForSeconds(0f);
						}
					}

					Progress = 0f;
				}

				S.Close();
				X.Finish();
			}

			// Free memory once per category instead of once per clip
			// (per-clip UnloadUnusedAssetsImmediate was the main slowdown).
			EditorUtility.UnloadUnusedAssetsImmediate();
			Resources.UnloadUnusedAssets();
			GC.Collect();
		}

		// Restore the clip that was loaded before export (while Save is still
		// false, so restoring does not write the last-exported clip to disk).
		Editor.Mirroring = editorMirroring;
		Editor.LoadData(editorData);

		Editor.Save = editorSave;
		Editor.SetTargetFramerate(editorRate);
		Editor.SetRandomSeed(editorSeed);
		Editor.IsPoint = editorIsPoint;
		Editor.IsTrain = editorIsTrain;
		Editor.Folder = editorFolder;

		StatusLabel = string.Empty;
		Exporting = false;
		yield return new WaitForSeconds(0f);

		Debug.Log("Exported " + total + " samples to " + outputRoot + ".");
	}


	public class Sample
	{
		public Frame Frame;
		public Matrix4x4 Head;
		public Vector3 Gaze;
		public Matrix4x4 RightHand;
		public Vector3 RightHand_IndexFingerTip;
		public Matrix4x4 LeftHand;
		public Vector3 LeftHand_IndexFingerTip;
		public Matrix4x4 Root;
		public string Head_Target_name;
		public string RightHand_Target_name;
		public string LeftHand_Target_name;
		public float Gaze_Phase = 0;
		public float DeicticGesture_Phase = 0;
		public float Idle_Phase = 0;

		public Sample(MotionEditor editor, float timestamp)
		{
			editor.LoadFrame(timestamp);
			Frame = editor.GetCurrentFrame();
			Root = Matrix4x4.TRS(editor.Root.position, editor.Root.rotation, Vector3.one);
			Head = Matrix4x4.TRS(editor.Head.position, editor.Head.rotation, Vector3.one);
			Gaze = editor.Gaze.forward;
			RightHand = Matrix4x4.TRS(editor.RightHand.position, editor.RightHand.rotation, Vector3.one);
			RightHand_IndexFingerTip = editor.RightHand_IndexFingerTip.position;
			LeftHand = Matrix4x4.TRS(editor.LeftHand.position, editor.LeftHand.rotation, Vector3.one);
			LeftHand_IndexFingerTip = editor.LeftHand_IndexFingerTip.position;
			Head_Target_name = ((TargetModule)editor.GetData().GetModule(Module.ID.Target)).Head_Target.Props[Frame.Index - 1];
			if (!editor.Mirroring)
			{
				RightHand_Target_name = ((TargetModule)editor.GetData().GetModule(Module.ID.Target)).RightHand_Target.Props[Frame.Index - 1];
				LeftHand_Target_name = ((TargetModule)editor.GetData().GetModule(Module.ID.Target)).LeftHand_Target.Props[Frame.Index - 1];
			}
			else
			{
				RightHand_Target_name = ((TargetModule)editor.GetData().GetModule(Module.ID.Target)).LeftHand_Target.Props[Frame.Index - 1];
				LeftHand_Target_name = ((TargetModule)editor.GetData().GetModule(Module.ID.Target)).RightHand_Target.Props[Frame.Index - 1];
			}
		}
	}
}
#endif