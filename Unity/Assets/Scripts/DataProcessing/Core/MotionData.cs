#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;

public class MotionData : ScriptableObject {

	public Frame[] Frames = new Frame[0];
	public Module[] Modules = new Module[0];
	public Sequence[] Sequences = new Sequence[0];

	public float Framerate = 1f;
	public Vector3 Offset = Vector3.zero;
	public float Scale = 1f;

	public bool Export = true;

	public void InspectAll(bool value) 
	{
		foreach(Module module in Modules) 
		{
			module.Inspect = value;
		}
	}

	public void VisualiseAll(bool value) 
	{
		foreach(Module module in Modules) 
		{
			module.Visualise = value;
		}
	}

	public string GetName() 
	{
		return name;
	}
	
	public float GetTotalTime() 
	{
		return Frames.Length / Framerate;
	}

	public int GetTotalFrames() {
		return Frames.Length;

	}

	public Frame GetFrame(int index) 
	{
		return Frames[Mathf.Clamp(index-1, 0, Frames.Length-1)];
	}

	public Frame GetFrame(float time) 
	{
		return Frames[Mathf.Clamp(Mathf.RoundToInt(time * Framerate), 0, Frames.Length-1)];
		/*
		if(time < 0f || time > GetTotalTime()) {
			Debug.Log("Please specify a time between 0 and " + GetTotalTime() + ". Given " + time + ".");
			return null;
		}
		return GetFrame(Mathf.Min(Mathf.RoundToInt(time * Framerate) + 1, GetTotalFrames()));
		*/
	}

	public Frame[] GetFrames(int start, int end) 
	{
		if(start < 1 || end > GetTotalFrames()) 
		{
			Debug.Log("Please specify indices between 1 and " + GetTotalFrames() + ". Given " + start + " and " + end + ".");
			return null;
		}
		int count = end-start+1;
		Frame[] frames = new Frame[count];
		for(int i=start; i<=end; i++) 
		{
			frames[i-start] = Frames[i-1];
		}
		return frames;
	}

	public Frame[] GetFrames(float start, float end) 
	{
		if(start < 0f || end > GetTotalTime()) 
		{
			Debug.Log("Please specify times between 0 and " + GetTotalTime() + ". Given " + start + " and " + end + ".");
			return null;
		}
		return GetFrames(GetFrame(start).Index, GetFrame(end).Index);
	}

	public void AddSequence() 
	{
		ArrayExtensions.Add(ref Sequences, new Sequence(1, GetTotalFrames()));
	}

	public void RemoveSequence(Sequence sequence) 
	{
		ArrayExtensions.Remove(ref Sequences, sequence);
	}

	public bool ContainedInSequences(Frame frame) 
	{
		foreach(Sequence s in Sequences) 
		{
			if(s.Contains(frame.Index)) 
			{
				return true;
			}
		}
		return false;
	}

	public Sequence GetUnrolledSequence() 
	{
		if(Sequences.Length == 0) 
		{
			return new Sequence(1, Frames.Length);
		}
		if(Sequences.Length == 1) 
		{
			return Sequences[0];
		}
		int start = Frames.Length;
		int end = 1;
		foreach(Sequence seq in Sequences) 
		{
			start = Mathf.Min(seq.Start, start);
			end = Mathf.Max(seq.End, end);
		}
		return new Sequence(start, end);
	}

	public void Load() {
		//Check Naming
		if(name == "Data") 
		{
			Debug.Log("Updating name of asset at " + AssetDatabase.GetAssetPath(this) + ".");
			string dataName = Directory.GetParent(AssetDatabase.GetAssetPath(this)).Name;
			int dataDot = dataName.LastIndexOf(".");
			if(dataDot != -1) 
			{
				dataName = dataName.Substring(0, dataDot);
			}
			AssetDatabase.RenameAsset(AssetDatabase.GetAssetPath(this), dataName);
			string parentDirectory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(this));
			int parentDot = parentDirectory.LastIndexOf(".");
			if(parentDot != -1) 
			{
				AssetDatabase.MoveAsset(parentDirectory, parentDirectory.Substring(0, parentDot));
			}
		}

		//Check modules
		for(int i=0; i<Modules.Length; i++) 
		{
			if(Modules[i] == null) 
			{
				Debug.Log("Removing missing module in " + GetName() + ".");
				ArrayExtensions.RemoveAt(ref Modules, i);
				i--;
			}
		}

		//Open Scene
		GetScene();
	}

	public void Unload() {
		Scene scene = EditorSceneManager.GetSceneByName(name);
		if(Application.isPlaying)
		{
			SceneManager.UnloadSceneAsync(scene);
		} else
		{
			// Close synchronously so a stale (unloaded) scene with the same name
			// cannot linger and be returned by GetScene() during batch export.
			EditorSceneManager.CloseScene(scene, true);
			EditorApplication.RepaintHierarchyWindow();
		}
	}

	public void Save() 
	{
		if(!Application.isPlaying) 
		{		
			EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetSceneByName(name)); 
			EditorSceneManager.SaveScene(EditorSceneManager.GetSceneByName(name));
			EditorUtility.SetDirty(this);
			AssetDatabase.SaveAssets();
			AssetDatabase.Refresh();
		}
	}

	public Scene GetScene() {
		for(int i=0; i<SceneManager.sceneCount; i++) {
			if(SceneManager.GetSceneAt(i).name == name && SceneManager.GetSceneAt(i).isLoaded) {
				return SceneManager.GetSceneAt(i);
			}
		}
		if(Application.isPlaying) {
			if(File.Exists(GetScenePath())) {
				EditorSceneManager.LoadSceneInPlayMode(GetScenePath(), new LoadSceneParameters(LoadSceneMode.Additive));
			} else {
				Debug.Log("Creating temporary scene for data " + name + ".");
				SceneManager.CreateScene(name);			}
		} else {
			Scene active = EditorSceneManager.GetActiveScene();
			if(File.Exists(GetScenePath())) {
				EditorSceneManager.OpenScene(GetScenePath(), OpenSceneMode.Additive);
			} else {
				Debug.Log("Recreating scene for data " + name + ".");
				EditorSceneManager.SaveScene(EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive), GetScenePath());
			}
			EditorSceneManager.SetActiveScene(SceneManager.GetSceneByName(name));
			Lightmapping.bakedGI = false;
			Lightmapping.realtimeGI = false;
			EditorSceneManager.SetActiveScene(active);
		}
		return SceneManager.GetSceneByName(name);
	}

	private string GetScenePath() 
	{
		return Path.GetDirectoryName(AssetDatabase.GetAssetPath(this)) + "/" + name + ".unity";
	}

	public void CreateScene() 
	{
		UnityEngine.SceneManagement.Scene active = EditorSceneManager.GetActiveScene();
		UnityEngine.SceneManagement.Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
		EditorSceneManager.SetActiveScene(scene);
		Lightmapping.bakedGI = false;
		Lightmapping.realtimeGI = false;
		EditorSceneManager.SetActiveScene(active);
		EditorSceneManager.SaveScene(scene, GetScenePath());
		EditorSceneManager.CloseScene(scene, true);
	}

	public Module AddModule(Module.ID type) 
	{
		Module module = System.Array.Find(Modules, x => x.GetID() == type);
		if(module != null) 
		{
			Debug.Log("Module of type " + type + " already exists in " + GetName() + ".");
		} 
		else 
		{
			string id = type + "Module";
			module = (Module)ScriptableObject.CreateInstance(id);
			if(module == null) 
			{
				Debug.Log("Module of class type " + id + " could not be loaded in " + GetName() + ".");
			} 
			else 
			{
				ArrayExtensions.Add(ref Modules, module.Initialise(this));
				AssetDatabase.AddObjectToAsset(Modules.Last(), this);
			}
		}
		return module;
	}

	public void RemoveModule(Module.ID type) 
	{
		Module module = GetModule(type);
		if(!module) 
		{
			Debug.Log("Module of type " + type + " does not exist in " + GetName() + ".");
		} 
		else 
		{
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		}
	}

	public void RemoveAllModules() 
	{
		foreach(Module module in Modules) 
		{
			ArrayExtensions.Remove(ref Modules, module);
			Utility.Destroy(module);
		}
	}

	public Module GetModule(Module.ID type) 
	{
		for(int i=0; i<Modules.Length; i++) 
		{
			if(Modules[i].GetID() == type) 
			{
				return Modules[i];
			}
		}
		//Debug.Log("Module of type " + type + " does not exist in " + GetName() + ".");
		return null;
	}
}
#endif