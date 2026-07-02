#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;

public class MotionImporter : EditorWindow
{
	public static EditorWindow Window;
	public static Vector2 Scroll;

	public enum Split { Train, Test }
	public Split ImportSplit = Split.Train;

	private string[] Subjects = new string[0];
	private int SubjectIndex = 0;

	private class Category
	{
		public string Name;
		public string SourcePath;
		public bool Enabled = true;
		public bool Foldout = false;
		public List<File> Files = new List<File>();
	}
	private List<Category> Categories = new List<Category>();

	private bool Initialized = false;
	private string StatusLabel = string.Empty;
	private bool Importing = false;

	// Raw motion capture data lives outside Assets/, next to it (Unity/Motion Capture Data).
	private static string RawRoot
	{
		get { return Directory.GetParent(Application.dataPath).FullName.Replace("\\", "/") + "/Motion Capture Data"; }
	}
	private const string DataRoot = "Assets/Data";

	[MenuItem("Motion Data Processing/Motion Importer")]
	static void Init()
	{
		Window = EditorWindow.GetWindow(typeof(MotionImporter));
		Scroll = Vector3.zero;
	}

	void OnGUI()
	{
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

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
					EditorGUILayout.LabelField("Data Importer");
				}

				if (!Initialized)
				{
					Load();
				}

				Split newSplit = (Split)EditorGUILayout.EnumPopup("Split", ImportSplit);
				if (newSplit != ImportSplit)
				{
					ImportSplit = newSplit;
					SubjectIndex = 0;
					Load();
				}

				if (ImportSplit == Split.Test)
				{
					if (Subjects.Length == 0)
					{
						EditorGUILayout.LabelField("No test subjects found under " + RawRoot + "/Test.");
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

				Utility.SetGUIColor(UltiDraw.White);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Source: " + GetSourceRoot());
					EditorGUILayout.LabelField("Destination: " + GetDestinationRoot());
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
										for (int i = 0; i < cat.Files.Count; i++) cat.Files[i].Import = true;
									}
									if (Utility.GUIButton("None", UltiDraw.DarkGrey, UltiDraw.White))
									{
										for (int i = 0; i < cat.Files.Count; i++) cat.Files[i].Import = false;
									}
									EditorGUILayout.EndHorizontal();

									for (int i = 0; i < cat.Files.Count; i++)
									{
										EditorGUILayout.BeginHorizontal();
										EditorGUILayout.LabelField((i + 1) + " - " + cat.Files[i].fileInfo.Name, GUILayout.Width(220f));
										GUILayout.FlexibleSpace();
										if (Utility.GUIButton("O", cat.Files[i].Import ? UltiDraw.DarkGreen : UltiDraw.DarkRed, UltiDraw.White, 50f))
										{
											cat.Files[i].Import = !cat.Files[i].Import;
										}
										EditorGUILayout.EndHorizontal();
									}
								}
							}
						}
					}
				}

				if (!Importing)
				{
					if (Utility.GUIButton("Reload", UltiDraw.DarkGrey, UltiDraw.White))
					{
						Load();
					}
					if (Utility.GUIButton("Import Motion", UltiDraw.DarkGrey, UltiDraw.White))
					{
						this.StartCoroutine(ImportMotion());
					}
				}
				else
				{
					EditorGUILayout.LabelField("Importing: " + StatusLabel);
					if (Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White))
					{
						this.StopAllCoroutines();
						Importing = false;
					}
				}
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private string GetSourceRoot()
	{
		if (ImportSplit == Split.Train)
		{
			return RawRoot + "/Train";
		}
		string subject = (Subjects.Length > 0 && SubjectIndex < Subjects.Length) ? Subjects[SubjectIndex] : "<subject>";
		return RawRoot + "/Test/" + subject;
	}

	private string GetDestinationRoot()
	{
		if (ImportSplit == Split.Train)
		{
			return DataRoot + "/Train";
		}
		string subject = (Subjects.Length > 0 && SubjectIndex < Subjects.Length) ? Subjects[SubjectIndex] : "<subject>";
		return DataRoot + "/Test/" + subject;
	}

	private void RefreshSubjects()
	{
		List<string> names = new List<string>();
		string testRoot = RawRoot + "/Test";
		if (Directory.Exists(testRoot))
		{
			foreach (DirectoryInfo d in new DirectoryInfo(testRoot).GetDirectories())
			{
				names.Add(d.Name);
			}
			names.Sort(StringComparer.Ordinal);
		}
		Subjects = names.ToArray();
		if (SubjectIndex >= Subjects.Length)
		{
			SubjectIndex = 0;
		}
	}

	public void Load()
	{
		Initialized = true;
		Categories = new List<Category>();

		if (ImportSplit == Split.Test)
		{
			RefreshSubjects();
		}

		string srcRoot = GetSourceRoot();
		if (!Directory.Exists(srcRoot))
		{
			return;
		}

		DirectoryInfo[] catDirs = new DirectoryInfo(srcRoot).GetDirectories();
		Array.Sort(catDirs, (a, b) => string.CompareOrdinal(a.Name, b.Name));
		foreach (DirectoryInfo dir in catDirs)
		{
			Category cat = new Category();
			cat.Name = dir.Name;
			cat.SourcePath = dir.FullName.Replace("\\", "/");

			FileInfo[] csvs = dir.GetFiles("*.csv");
			Array.Sort(csvs, (a, b) => NaturalCompare(a.Name, b.Name));
			foreach (FileInfo fi in csvs)
			{
				File file = new File();
				file.fileInfo = fi;
				file.Import = true;
				cat.Files.Add(file);
			}
			cat.Enabled = cat.Files.Count > 0;
			Categories.Add(cat);
		}
	}

	private static int CountSelected(Category cat)
	{
		int n = 0;
		for (int i = 0; i < cat.Files.Count; i++)
		{
			if (cat.Files[i].Import)
			{
				n += 1;
			}
		}
		return n;
	}

	private static int NaturalCompare(string a, string b)
	{
		int ai, bi;
		bool an = int.TryParse(Path.GetFileNameWithoutExtension(a), out ai);
		bool bn = int.TryParse(Path.GetFileNameWithoutExtension(b), out bi);
		if (an && bn)
		{
			return ai.CompareTo(bi);
		}
		return string.CompareOrdinal(a, b);
	}

	private float DeriveFramerate(string categoryName)
	{
		// The "Referencing multiple targets with diverse gestures" repertoires were
		// captured at 90 fps; every other repertoire at 60 fps.
		if (ImportSplit == Split.Train)
		{
			if (categoryName == "Referencing multiple targets with diverse gestures (pointing session 2)" ||
				categoryName == "Referencing multiple targets with diverse gestures (touching)")
			{
				return 90f;
			}
		}
		else
		{
			if (categoryName == "Referencing multiple targets with diverse gestures (pointing)" ||
				categoryName == "Referencing multiple targets with diverse gestures (touching)")
			{
				return 90f;
			}
		}
		return 60f;
	}

	private static void EnsureFolder(string assetPath)
	{
		assetPath = assetPath.Replace("\\", "/");
		if (AssetDatabase.IsValidFolder(assetPath))
		{
			return;
		}
		int slash = assetPath.LastIndexOf('/');
		string parent = assetPath.Substring(0, slash);
		string leaf = assetPath.Substring(slash + 1);
		EnsureFolder(parent);
		AssetDatabase.CreateFolder(parent, leaf);
	}

	private IEnumerator ImportMotion()
	{
		Importing = true;
		string destinationRoot = GetDestinationRoot();

		for (int c = 0; c < Categories.Count; c++)
		{
			if (!Importing)
			{
				break;
			}
			Category cat = Categories[c];
			if (!cat.Enabled || CountSelected(cat) == 0)
			{
				continue;
			}

			string destination = destinationRoot + "/" + cat.Name;
			EnsureFolder(destination);
			float framerate = DeriveFramerate(cat.Name);

			for (int f = 0; f < cat.Files.Count; f++)
			{
				if (!Importing)
				{
					break;
				}
				if (!cat.Files[f].Import)
				{
					continue;
				}

				string fileName = Path.GetFileNameWithoutExtension(cat.Files[f].fileInfo.Name);
				StatusLabel = cat.Name + " / " + fileName;

				if (!Directory.Exists(destination + "/" + fileName))
				{
					AssetDatabase.CreateFolder(destination, fileName);
					MotionData data = ScriptableObject.CreateInstance<MotionData>();
					data.name = fileName;
					AssetDatabase.CreateAsset(data, destination + "/" + fileName + "/" + data.name + ".asset");

					string[] lines = System.IO.File.ReadAllLines(cat.Files[f].fileInfo.FullName);
					char[] commaspace = new char[] { ',' };

					ArrayExtensions.Resize(ref data.Frames, lines.Length);
					data.Framerate = framerate;

					for (int k = 0; k < data.GetTotalFrames(); k++)
					{
						data.Frames[k] = new Frame(data, k + 1, (float)k / data.Framerate);
						string[] entries = lines[k].Split(commaspace);
						Quaternion rot;
						Vector3 pos;

						// Head
						rot = Quaternion.LookRotation(new Vector3(FileUtility.ReadFloat(entries[0]), FileUtility.ReadFloat(entries[1]), FileUtility.ReadFloat(entries[2])),
													  new Vector3(FileUtility.ReadFloat(entries[3]), FileUtility.ReadFloat(entries[4]), FileUtility.ReadFloat(entries[5])));
						pos = new Vector3(FileUtility.ReadFloat(entries[6]), FileUtility.ReadFloat(entries[7]), FileUtility.ReadFloat(entries[8]));
						Matrix4x4 Head = Matrix4x4.TRS(pos, rot, Vector3.one);
						// GazeOrigin
						Vector3 GazeOrigin = new Vector3(FileUtility.ReadFloat(entries[9]), FileUtility.ReadFloat(entries[10]), FileUtility.ReadFloat(entries[11]));
						// GazeDirection
						Vector3 GazeDirection = new Vector3(FileUtility.ReadFloat(entries[12]), FileUtility.ReadFloat(entries[13]), FileUtility.ReadFloat(entries[14]));
						// RightHand
						rot = Quaternion.LookRotation(new Vector3(FileUtility.ReadFloat(entries[15]), FileUtility.ReadFloat(entries[16]), FileUtility.ReadFloat(entries[17])),
													  new Vector3(FileUtility.ReadFloat(entries[18]), FileUtility.ReadFloat(entries[19]), FileUtility.ReadFloat(entries[20])));
						pos = new Vector3(FileUtility.ReadFloat(entries[21]), FileUtility.ReadFloat(entries[22]), FileUtility.ReadFloat(entries[23]));
						Matrix4x4 RightHand = Matrix4x4.TRS(pos, rot, Vector3.one);
						// RightHand_IndexFingerTip
						Vector3 RightHand_IndexFingerTip = new Vector3(FileUtility.ReadFloat(entries[24]), FileUtility.ReadFloat(entries[25]), FileUtility.ReadFloat(entries[26]));
						// LeftHand
						rot = Quaternion.LookRotation(new Vector3(FileUtility.ReadFloat(entries[27]), FileUtility.ReadFloat(entries[28]), FileUtility.ReadFloat(entries[29])),
													  new Vector3(FileUtility.ReadFloat(entries[30]), FileUtility.ReadFloat(entries[31]), FileUtility.ReadFloat(entries[32])));
						pos = new Vector3(FileUtility.ReadFloat(entries[33]), FileUtility.ReadFloat(entries[34]), FileUtility.ReadFloat(entries[35]));
						Matrix4x4 LeftHand = Matrix4x4.TRS(pos, rot, Vector3.one);
						// LeftHand_IndexFingerTip
						Vector3 LeftHand_IndexFingerTip = new Vector3(FileUtility.ReadFloat(entries[36]), FileUtility.ReadFloat(entries[37]), FileUtility.ReadFloat(entries[38]));
						// Targets
						Vector3 Target1 = new Vector3(FileUtility.ReadFloat(entries[39]), FileUtility.ReadFloat(entries[40]), FileUtility.ReadFloat(entries[41]));
						Vector3 Target2 = new Vector3(FileUtility.ReadFloat(entries[42]), FileUtility.ReadFloat(entries[43]), FileUtility.ReadFloat(entries[44]));
						Vector3 Target3 = new Vector3(FileUtility.ReadFloat(entries[45]), FileUtility.ReadFloat(entries[46]), FileUtility.ReadFloat(entries[47]));

						data.Frames[k].Head = Head;
						data.Frames[k].GazeOrigin = GazeOrigin;
						data.Frames[k].GazeDirection = GazeDirection;
						data.Frames[k].RightHand = RightHand;
						data.Frames[k].RightHand_IndexFingerTip = RightHand_IndexFingerTip;
						data.Frames[k].LeftHand = LeftHand;
						data.Frames[k].LeftHand_IndexFingerTip = LeftHand_IndexFingerTip;
						data.Frames[k].Targets[0] = Target1;
						data.Frames[k].Targets[1] = Target2;
						data.Frames[k].Targets[2] = Target3;
					}

					// Add Scene
					data.CreateScene();
					data.AddSequence();

					// Save
					EditorUtility.SetDirty(data);
				}
				else
				{
					Debug.Log("File with name " + fileName + " already exists in " + destination + ".");
				}
				yield return new WaitForSeconds(0f);
			}
		}

		AssetDatabase.SaveAssets();
		AssetDatabase.Refresh();

		StatusLabel = string.Empty;
		Importing = false;
		yield return new WaitForSeconds(0f);
	}

	[System.Serializable]
	public class File
	{
		public FileInfo fileInfo;
		public bool Import;
	}
}
#endif
