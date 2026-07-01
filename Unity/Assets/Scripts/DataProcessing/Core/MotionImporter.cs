#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

public class MotionImporter : EditorWindow {
	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Source = "Record/Train/Point_One";
	public string Destination = "Assets/Data/Train/Point_One";
	public string Filter = string.Empty;
	public File[] Files = new File[0];
	public File[] Instances = new File[0];
	public bool Importing = false;

	private int Start = 0;
	private int End = 0;

	public int Page = 1;
	public const int Items = 25;

	[MenuItem ("Motion Data Processing/Motion Importer")]
	static void Init() 
	{
		Window = EditorWindow.GetWindow(typeof(MotionImporter));
		Scroll = Vector3.zero;
	}
	
	void OnGUI() 
	{
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) 
		{
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) 
			{
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) 
				{
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Data Importer");
				}
		
				if(!Importing) 
				{
					if(Utility.GUIButton("Import Motion", UltiDraw.DarkGrey, UltiDraw.White)) 
					{
						this.StartCoroutine(ImportMotion());
					}
				} 
				else 
				{
					if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) 
					{
						this.StopAllCoroutines();
						Importing = false;
					}
				}

				using(new EditorGUILayout.VerticalScope ("Box")) 
				{
					EditorGUILayout.LabelField("Source");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					Source = EditorGUILayout.TextField(Source);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) 
					{
						Source = EditorUtility.OpenFolderPanel("Motion Importer", Source == string.Empty ? Application.dataPath : Source, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.LabelField("Destination");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					Destination = EditorGUILayout.TextField(Destination);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if (GUILayout.Button("O", GUILayout.Width(20)))
					{
						Destination = EditorUtility.OpenFolderPanel("Motion Importer", Destination == string.Empty ? Application.dataPath : Destination, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					string filter = EditorGUILayout.TextField("Filter", Filter);
					if (Filter != filter)
					{
						Filter = filter;
						ApplyFilter();
					} 

					if (Utility.GUIButton("Load Directory", UltiDraw.DarkGrey, UltiDraw.White)) 
					{
						LoadDirectory();
					}

					int start = (Page-1)*Items;
					int end = Mathf.Min(start+Items, Instances.Length);
					int pages = Mathf.CeilToInt(Instances.Length/Items)+1;
					Utility.SetGUIColor(UltiDraw.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) 
					{
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) 
						{
							Page = Mathf.Max(Page-1, 1);
						}
						EditorGUILayout.LabelField("Page " + Page + "/" + pages);
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) 
						{
							Page = Mathf.Min(Page+1, pages);
						}
						EditorGUILayout.EndHorizontal();
					}

					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White)) 
					{
						for(int i=0; i<Instances.Length; i++) 
						{
							Instances[i].Import = true;
						}
					}
					if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White)) 
					{
						for(int i=0; i<Instances.Length; i++) 
						{
							Instances[i].Import = false;
						}
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.BeginHorizontal();
					Start = EditorGUILayout.IntField("Start", Start);
					End = EditorGUILayout.IntField("End", End);
					if (Utility.GUIButton("Toggle", UltiDraw.DarkGrey, UltiDraw.White))
					{
						for (int i = Start - 1; i <= End - 1; i++)
						{
							Instances[i].Import = !Instances[i].Import;
						}
					}
					EditorGUILayout.EndHorizontal(); 

					for (int i=start; i<end; i++) 
					{
						if(Instances[i].Import) 
						{
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} 
						else 
						{
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) 
						{
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Instances[i].Import = EditorGUILayout.Toggle(Instances[i].Import, GUILayout.Width(20f));
							EditorGUILayout.LabelField(Instances[i].fileInfo.Name);
							EditorGUILayout.EndHorizontal();							
						}
					}

					/*
					if(Utility.GUIButton("DeBug", UltiDraw.DarkGrey, UltiDraw.White)) 
					{
						DeBug(); 
					}
					*/
				}		
			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void DeBug()
	{
		#region Debug
		Debug.Log("Debug!");
		#endregion Region						
	}

	private void LoadDirectory() 
	{
		if(Directory.Exists(Source)) 
		{			
			DirectoryInfo info = new DirectoryInfo(Source);
			// FileInfo[] items = info.GetFiles("*.csv");				
			FileInfo[] items = info.GetFiles("*.csv").OrderBy(file => int.Parse(file.Name.Split('.')[0])).ToArray();
			Files = new File[items.Length];
			Debug.Log(items.Length);
			for (int i=0; i< items.Length; i++) 
			{
				Files[i] = new File();
				Files[i].fileInfo = items[i];
				Files[i].Import = true;
			}
		} 
		else 
		{
			Files = new File[0];
		}
		ApplyFilter();
		Page = 1;
		Debug.Log("Load Directory!");
	}

	private void ApplyFilter()
	{
		if (Filter == string.Empty)
		{
			Instances = Files;
		}
		else
		{
			List<File> instances = new List<File>();
			for (int i = 0; i < Files.Length; i++)
			{
				if (Files[i].fileInfo.Name.ToLowerInvariant().Contains(Filter.ToLowerInvariant()))
				{
					instances.Add(Files[i]);
				}
			}
			Instances = instances.ToArray();
		}
	} 

	private IEnumerator ImportMotion() 
	{
		if(!AssetDatabase.IsValidFolder(Destination)) 
		{
			Debug.Log("Folder " + "'" + Destination + "'" + " is not valid.");
		}
		else 
		{
			Importing = true;
			for(int f=0; f<Files.Length; f++) 
			{
				if(Files[f].Import) 
				{
					string fileName = Files[f].fileInfo.Name.Replace(".bvh", "");
					if(!Directory.Exists(Destination + "/"+fileName))
					{
						AssetDatabase.CreateFolder(Destination, fileName);
						MotionData data = ScriptableObject.CreateInstance<MotionData>();
						data.name = fileName;
						AssetDatabase.CreateAsset(data, Destination + "/"+fileName+"/"+data.name+".asset");

						string[] lines = System.IO.File.ReadAllLines(Files[f].fileInfo.FullName);
						char[] commaspace = new char[] {','};

						ArrayExtensions.Resize(ref data.Frames, lines.Length);
						// data.Framerate = 60f;
						data.Framerate = 90f;

						for (int k=0; k<data.GetTotalFrames(); k++) 
						{
							data.Frames[k] = new Frame(data, k+1, (float)k / data.Framerate);
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

							data.Frames[k].Head= Head;
							data.Frames[k].GazeOrigin = GazeOrigin;
							data.Frames[k].GazeDirection = GazeDirection;
							data.Frames[k].RightHand= RightHand;
							data.Frames[k].RightHand_IndexFingerTip = RightHand_IndexFingerTip;
							data.Frames[k].LeftHand = LeftHand;
							data.Frames[k].LeftHand_IndexFingerTip = LeftHand_IndexFingerTip;
							data.Frames[k].Targets[0] = Target1;
							data.Frames[k].Targets[1] = Target2;
							data.Frames[k].Targets[2] = Target3;
						}

						//Add Scene
						data.CreateScene();
						data.AddSequence();

						//Save
						EditorUtility.SetDirty(data);
						Debug.Log(f+"th Imported!");
					} 
					else 
					{
						Debug.Log("File with name " + fileName + " already exists.");
					}
					yield return new WaitForSeconds(0f);
				}
			}
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
			Importing = false;
		}
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