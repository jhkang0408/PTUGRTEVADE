#if UNITY_EDITOR
using NetMQ;
using NetMQ.Sockets;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

// Real-time inference client for the Inference scene.
//
// The user's 3-point tracking (Head / RightHand / LeftHand, plus gaze and index finger tips) is played back by the
// MotionEditor. Every frame the current window of features is sent to DeepLearning/inference.py (ZMQ REQ/REP), which
// runs TENet (target estimation) and BP-GTNet (part-wise translation) and returns the avatar's 3-point pose together
// with the estimated targets. The avatar pose is written straight into the Avatar's Head / Gaze / RightHand /
// LeftHand / IndexFingerTip transforms and drawn with UltiDraw. No character mesh, IK or post-processing is involved.
public class Inference : MonoBehaviour
{
	// A character represented only by its tracked points (all children of the character root).
	private class Points
	{
		public Transform Head, Gaze, RightHand, RightHand_IndexFingerTip, LeftHand, LeftHand_IndexFingerTip;

		public Points(GameObject root)
		{
			Head = Find(root, "Head");
			Gaze = Find(root, "Gaze");
			RightHand = Find(root, "RightHand");
			RightHand_IndexFingerTip = Find(root, "RightHand_IndexFingerTip");
			LeftHand = Find(root, "LeftHand");
			LeftHand_IndexFingerTip = Find(root, "LeftHand_IndexFingerTip");
		}

		private static Transform Find(GameObject root, string name)
		{
			Transform t = root.transform.Find(name);
			if (t == null)
			{
				Debug.LogError(root.name + " has no child named '" + name + "'.");
			}
			return t;
		}
	}

	[Serializable]
	public class Target
	{
		public Vector3 cartesian;
		public Vector3 spherical; // (horizontal angle, vertical angle, distance) relative to the user's root
		public Vector3 direction;
		public string id;
	}

	[Header("Server")]
	public string ServerAddress = "tcp://localhost:12345";
	public float ReceiveTimeout = 5f; // seconds to wait for a reply before reconnecting

	[Header("Motion Editor")]
	public MotionEditor editor = null;

	[Header("User (3-point tracking input)")]
	public GameObject User;
	[Range(0.8f, 1.2f)]
	public float User_Scale = 1f;

	[Header("Avatar (3-point tracking output)")]
	public GameObject Avatar;
	[Range(0.8f, 1.2f)]
	public float Avatar_Scale = 1f;
	public Vector3 Avatar_Root_Offset = new Vector3(10f, 0f, 0f); // avatar root = user root + this offset (height is kept)

	[Header("Targets")]
	public GameObject[] Targets_GameObjects_User = new GameObject[3];     // named "User_Target1" ... (follow the MotionEditor targets)
	public GameObject[] Targets_GameObjects_Avatar = new GameObject[3];   // named "Avatar_Target1" ... (the avatar's dissimilar environment)
	public bool IsThirdTargetFix = true;  // true: User_Target3 follows the clip's Target3; false: place User_Target3 manually in the scene
	public List<Target> Targets_User = new List<Target>();

	[Header("Control")]
	public bool Play = false;            // start streaming to the inference server
	public bool Reset = true;            // stop streaming
	public bool CopyUserMotion = false;  // mirror the user's pose onto the avatar (no network)
	public bool SetGTTarget = false;     // use the labeled targets instead of the estimated ones
	public int Window_Length_TENet = 45;
	public int Window_Length_BP_GTNet = 30;
	public int Window_Length_UserPose = 5;
	public float Idle_Progression_Threshold = 0.5f;

	[Header("Visualization")]
	public bool DrawRoot = false;
	public bool DrawTarget = true;
	public bool DrawEstimatedTarget = false;
	public bool DrawEstimatedTarget_Confidence = true;
	public bool DrawUserEE = true;
	public bool DrawAvatarEE = true;
	public bool DrawRay = false;
	public bool DrawRayTENet = false;
	public bool DrawFPS = true;
	public bool DrawCaptions = true;     // "User (Up)" / "Avatar (Front)" ... labels in the corner of each viewport

	/* Progressions returned by the network */
	private float Gaze_Progression = 0f;
	private float DeicticGesture_Progression_RightHand = 0f;
	private float DeicticGesture_Progression_LeftHand = 0f;
	private float Idle_Progression_Head = 0f;
	private float Idle_Progression_RightHand = 0f;
	private float Idle_Progression_LeftHand = 0f;

	/* Networking */
	private RequestSocket socket;
	private Thread inferenceThread;
	private AutoResetEvent processingEvent = new AutoResetEvent(true);
	private volatile bool isThreadRunning = false;
	private volatile bool isProcessing = false;
	private volatile bool hasNewOutput = false;
	private List<string> input_window_TENet = new List<string>();
	private List<string> input_window_BP_GTNet = new List<string>();
	private List<string> input_window_UserPose = new List<string>();
	private string currentInput_TENet = null;
	private string currentInput_BP_GTNet = null;
	private string currentInput_UserPose = null;
	private string currentInput_AvatarPose = null;
	private string currentInput_Combinations = null;
	private string currentOutput = null;
	private float inferenceTime = 0f;
	private float inferenceFPS = 0f;

	/* Characters */
	private Points user, avatar;
	private Matrix4x4 User_FixedRoot = Matrix4x4.identity;
	private Matrix4x4 Avatar_FixedRoot = Matrix4x4.identity;
	private Matrix4x4 User_Root = Matrix4x4.identity;
	private Vector3 User_RightHand_Default, User_LeftHand_Default;
	private Vector3 User_RightHand_IndexFingerTip_Default, User_LeftHand_IndexFingerTip_Default;
	private string previous_data_name = "";

	/* Targets */
	private Dictionary<string, Transform> userTargets = new Dictionary<string, Transform>();
	private Dictionary<string, Transform> avatarTargets = new Dictionary<string, Transform>();
	private int num_target;
	private float[] allCombinations = new float[0];
	private string Head_Target_Name, RightHand_Target_Name, LeftHand_Target_Name;
	private string Head_Target_Name_previous, RightHand_Target_Name_previous, LeftHand_Target_Name_previous;
	private Vector3 User_Head_Target, User_RightHand_Target, User_LeftHand_Target;
	private Vector3 Avatar_Head_Target, Avatar_RightHand_Target, Avatar_LeftHand_Target;
	private float Confidence = 0f;
	private float Confidence_Head = 0f;
	private float Confidence_RightHand = 0f;
	private float Confidence_LeftHand = 0f;
	private Vector3 gaze_detected, ef_righthand_detected, ef_lefthand_detected;
	private Vector3 gaze_corrected;

	private void Start()
	{
		user = new Points(User);
		avatar = new Points(Avatar);

		for (int i = 0; i < Targets_GameObjects_User.Length; i++)
		{
			userTargets[GetTargetId(Targets_GameObjects_User[i])] = Targets_GameObjects_User[i].transform;
		}
		for (int i = 0; i < Targets_GameObjects_Avatar.Length; i++)
		{
			avatarTargets[GetTargetId(Targets_GameObjects_Avatar[i])] = Targets_GameObjects_Avatar[i].transform;
		}
		num_target = Targets_GameObjects_User.Length + 1; // + TargetNull
		Targets_User = new List<Target>();
		for (int i = 0; i < num_target; i++)
		{
			Targets_User.Add(new Target());
		}
		Targets_User[num_target - 1].id = "TargetNull";

		Head_Target_Name = RightHand_Target_Name = LeftHand_Target_Name = "TargetNull";
		Head_Target_Name_previous = RightHand_Target_Name_previous = LeftHand_Target_Name_previous = "TargetNull";

		InitializeRoots();
		ApplySamePose();

		socket = new RequestSocket();
		TryConnect();
	}

	private static string GetTargetId(GameObject target)
	{
		// "User_Target1" -> "Target1"
		string[] parts = target.name.Split('_');
		return parts.Length > 1 ? parts[1] : target.name;
	}

	// Fixed roots (the frame the networks were trained in) and the user's rest pose, taken from the first frame of the clip.
	private void InitializeRoots()
	{
		MotionData data = editor != null ? editor.GetData() : null;
		if (data != null)
		{
			Frame first = data.GetFrame(0);
			User_FixedRoot = Matrix4x4.TRS(first.Head.GetPosition(), Quaternion.LookRotation(Vector3.ProjectOnPlane(first.Head.GetForward(), Vector3.up).normalized, Vector3.up), Vector3.one);
			User_RightHand_Default = first.RightHand.GetPosition();
			User_LeftHand_Default = first.LeftHand.GetPosition();
			User_RightHand_IndexFingerTip_Default = first.RightHand_IndexFingerTip;
			User_LeftHand_IndexFingerTip_Default = first.LeftHand_IndexFingerTip;
			previous_data_name = data.name;
		}
		else
		{
			User_FixedRoot = Matrix4x4.TRS(user.Head.position, Quaternion.LookRotation(Vector3.ProjectOnPlane(user.Head.forward, Vector3.up).normalized, Vector3.up), Vector3.one);
			User_RightHand_Default = user.RightHand.position;
			User_LeftHand_Default = user.LeftHand.position;
			User_RightHand_IndexFingerTip_Default = user.RightHand_IndexFingerTip.position;
			User_LeftHand_IndexFingerTip_Default = user.LeftHand_IndexFingerTip.position;
			previous_data_name = "";
		}
		Avatar_FixedRoot = Matrix4x4.TRS(User_FixedRoot.GetPosition() + Avatar_Root_Offset, Quaternion.identity, Vector3.one);

		User_Head_Target = User_RightHand_Target = User_LeftHand_Target = userTargets.Count > 0 ? userTargets.Values.First().position : User_FixedRoot.GetPosition();
		Avatar_Head_Target = Avatar_RightHand_Target = Avatar_LeftHand_Target = avatarTargets.Count > 0 ? avatarTargets.Values.First().position : Avatar_FixedRoot.GetPosition();
	}

	private void Update()
	{
		if (editor == null || editor.GetData() == null)
		{
			return;
		}

		SetTarget();

		if (Reset)
		{
			Play = false;
			Reset = false;
			StopInferenceThread();
		}

		if (CopyUserMotion)
		{
			StopInferenceThread();
			Play = false;
			ApplySamePose();
		}

		if (Play)
		{
			if (!isThreadRunning)
			{
				StartInferenceThread();
			}

			if (hasNewOutput && currentOutput != null)
			{
				SetOutput(currentOutput.Split(','));
				hasNewOutput = false;
			}

			if (processingEvent.WaitOne(0))
			{
				currentInput_TENet = SetInput("TENet", 12);
				currentInput_BP_GTNet = SetInput("BP_GTNet", 21);
				currentInput_UserPose = SetInput("UserPose", 27);
				currentInput_AvatarPose = SetInput("AvatarPose", 36);
				currentInput_Combinations = string.Join(", ", allCombinations.Select(f => f.ToString("F8")));
				isProcessing = true;
			}
		}
	}

	/* ---------------------------------------------------------------- Networking */

	private void StartInferenceThread()
	{
		if (isThreadRunning) return;
		isThreadRunning = true;
		inferenceThread = new Thread(InferenceLoop);
		inferenceThread.IsBackground = true;
		inferenceThread.Start();
	}

	private void StopInferenceThread()
	{
		isThreadRunning = false;
		if (inferenceThread != null && inferenceThread.IsAlive)
		{
			inferenceThread.Join(100);
		}
		input_window_TENet.Clear();
		input_window_BP_GTNet.Clear();
		input_window_UserPose.Clear();
		isProcessing = false;
		processingEvent.Set();
	}

	private void InferenceLoop()
	{
		while (isThreadRunning)
		{
			try
			{
				if (isProcessing && currentInput_TENet != null)
				{
					Push(input_window_TENet, currentInput_TENet, Window_Length_TENet);
					Push(input_window_BP_GTNet, currentInput_BP_GTNet, Window_Length_BP_GTNet);
					Push(input_window_UserPose, currentInput_UserPose, Window_Length_UserPose);

					if (input_window_TENet.Count == Window_Length_TENet)
					{
						System.Diagnostics.Stopwatch stopwatch = System.Diagnostics.Stopwatch.StartNew();

						string finalInput = string.Join(", ", input_window_TENet) + ", " +
						                    string.Join(", ", input_window_BP_GTNet) + ", " +
						                    string.Join(", ", input_window_UserPose) + ", " +
						                    currentInput_AvatarPose + ", " +
						                    currentInput_Combinations;

						string output;
						if (socket.TrySendFrame(TimeSpan.FromSeconds(ReceiveTimeout), finalInput) && socket.TryReceiveFrameString(TimeSpan.FromSeconds(ReceiveTimeout), out output))
						{
							currentOutput = output;
							hasNewOutput = true;
							stopwatch.Stop();
							inferenceTime = (float)stopwatch.Elapsed.TotalMilliseconds;
							inferenceFPS = 1000f / Mathf.Max(inferenceTime, 1e-3f);
						}
						else
						{
							Debug.LogWarning("No reply from the inference server at " + ServerAddress + ". Is DeepLearning/inference.py running?");
							ReconnectSocket();
						}
					}

					isProcessing = false;
					processingEvent.Set();
				}
				else
				{
					Thread.Sleep(1);
				}
			}
			catch (Exception e)
			{
				Debug.LogError("Error in inference thread: " + e);
				isThreadRunning = false;
				isProcessing = false;
				processingEvent.Set();
				break;
			}
		}
	}

	private static void Push(List<string> window, string value, int length)
	{
		window.Add(value);
		while (window.Count > length)
		{
			window.RemoveAt(0);
		}
	}

	private void TryConnect()
	{
		try
		{
			socket.Connect(ServerAddress);
			Debug.Log("Inference client connected to " + ServerAddress);
		}
		catch (NetMQException ex)
		{
			Debug.LogWarning("Inference client could not connect to " + ServerAddress + ": " + ex);
		}
	}

	// A REQ socket cannot send again after a lost reply, so replace it.
	private void ReconnectSocket()
	{
		try
		{
			socket.Close();
		}
		catch (Exception) { }
		socket = new RequestSocket();
		TryConnect();
	}

	private void OnDisable()
	{
		StopInferenceThread();
		if (socket != null)
		{
			socket.Close();
			socket = null;
		}
		NetMQConfig.Cleanup(false);
	}

	/* ---------------------------------------------------------------- Pose */

	// Copies the user's pose onto the avatar (relative to the two fixed roots).
	public void ApplySamePose()
	{
		avatar.Head.position = user.Head.position.GetRelativePositionTo(User_FixedRoot).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.Head.rotation = Quaternion.LookRotation(user.Head.forward.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot),
		                                               user.Head.up.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot));
		avatar.Gaze.position = avatar.Head.TransformPoint(user.Head.InverseTransformPoint(user.Gaze.position));
		avatar.Gaze.forward = avatar.Head.TransformDirection(user.Head.InverseTransformDirection(user.Gaze.forward));

		avatar.RightHand.position = user.RightHand.position.GetRelativePositionTo(User_FixedRoot).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.RightHand.rotation = Quaternion.LookRotation(user.RightHand.forward.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot),
		                                                    user.RightHand.up.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot));
		avatar.RightHand_IndexFingerTip.position = avatar.RightHand.TransformPoint(user.RightHand.InverseTransformPoint(user.RightHand_IndexFingerTip.position));

		avatar.LeftHand.position = user.LeftHand.position.GetRelativePositionTo(User_FixedRoot).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.LeftHand.rotation = Quaternion.LookRotation(user.LeftHand.forward.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot),
		                                                   user.LeftHand.up.GetRelativeDirectionTo(User_FixedRoot).GetRelativeDirectionFrom(Avatar_FixedRoot));
		avatar.LeftHand_IndexFingerTip.position = avatar.LeftHand.TransformPoint(user.LeftHand.InverseTransformPoint(user.LeftHand_IndexFingerTip.position));
	}

	/* ---------------------------------------------------------------- Targets */

	public void SetTarget()
	{
		Matrix4x4 Root = Matrix4x4.TRS(user.Head.position, User_FixedRoot.GetRotation(), Vector3.one);

		// The user's targets follow the clip (the third one only while IsThirdTargetFix is on);
		// the avatar's targets are placed freely in the scene.
		for (int i = 0; i < Targets_GameObjects_User.Length && i < editor.Targets.Length; i++)
		{
			if (i == 2 && !IsThirdTargetFix)
			{
				continue;
			}
			Targets_GameObjects_User[i].transform.position = editor.Targets[i].position;
		}

		for (int i = 0; i < Targets_GameObjects_User.Length; i++)
		{
			Vector3 position = Targets_GameObjects_User[i].transform.position;
			Vector3 direction = (position - Root.GetPosition()).normalized.GetRelativeDirectionTo(Root);
			float horizontal = Mathf.Atan2(direction.x, direction.z);
			float vertical = Mathf.Atan2(direction.y, Mathf.Sqrt(direction.x * direction.x + direction.z * direction.z));
			float distance = editor.IsPoint ? 1f : Mathf.Clamp((position - Root.GetPosition()).magnitude, 0f, 1f);
			Targets_User[i].cartesian = position;
			Targets_User[i].spherical = new Vector3(horizontal, vertical, distance);
			Targets_User[i].direction = direction;
			Targets_User[i].id = GetTargetId(Targets_GameObjects_User[i]);
		}
		Targets_User[num_target - 1].spherical = Vector3.zero;
		Targets_User[num_target - 1].id = "TargetNull";

		allCombinations = GenerateAllCombinations(Targets_User);

		if (Head_Target_Name != "TargetNull")
		{
			User_Head_Target = userTargets[Head_Target_Name].position;
			Avatar_Head_Target = avatarTargets[Head_Target_Name].position;
		}
		if (RightHand_Target_Name != "TargetNull")
		{
			User_RightHand_Target = userTargets[RightHand_Target_Name].position;
			Avatar_RightHand_Target = avatarTargets[RightHand_Target_Name].position;
		}
		if (LeftHand_Target_Name != "TargetNull")
		{
			User_LeftHand_Target = userTargets[LeftHand_Target_Name].position;
			Avatar_LeftHand_Target = avatarTargets[LeftHand_Target_Name].position;
		}
	}

	// Every (head, right hand, left hand) target assignment as 3 spherical coordinates each; the server scores them all.
	private float[] GenerateAllCombinations(List<Target> allTargets)
	{
		int totalCombinations = num_target * num_target * num_target;
		const int vectorDimension = 9;
		float[] combinations = new float[totalCombinations * vectorDimension];
		for (int i = 0; i < totalCombinations; i++)
		{
			int[] indices = DecodeIndexToTargetIndices(i);
			for (int slot = 0; slot < 3; slot++)
			{
				int baseIndex = i * vectorDimension + slot * 3;
				combinations[baseIndex + 0] = allTargets[indices[slot]].spherical.x;
				combinations[baseIndex + 1] = allTargets[indices[slot]].spherical.y;
				combinations[baseIndex + 2] = allTargets[indices[slot]].spherical.z;
			}
		}
		return combinations;
	}

	private int[] DecodeIndexToTargetIndices(int index)
	{
		int[] result = new int[3];
		int remaining = index;
		for (int slot = 2; slot >= 0; slot--)
		{
			result[slot] = remaining % num_target;
			remaining /= num_target;
		}
		return result;
	}

	/* ---------------------------------------------------------------- Network input */

	private bool IsLookingAtAnyTarget(Vector3 gaze)
	{
		for (int i = 0; i < Targets_GameObjects_User.Length; i++)
		{
			Vector3 ET = Targets_GameObjects_User[i].transform.position - user.Head.position;
			if (Targets_GameObjects_User[i].transform.position.y > 0 && Vector3.Angle(ET.normalized, gaze.normalized) < 10f)
			{
				return true;
			}
		}
		return false;
	}

	public float[] CalculateInput_TENet(int input_size)
	{
		float[] input_float = new float[input_size];

		// A new clip means a new rest pose.
		string current_data_name = editor.GetData().name;
		if (current_data_name != previous_data_name)
		{
			InitializeRoots();
			Debug.Log("Animation clip changed to " + current_data_name + ".");
		}
		previous_data_name = current_data_name;

		User_Root.SetTRS(user.Head.position, Quaternion.LookRotation(Vector3.ProjectOnPlane(user.Head.forward, Vector3.up).normalized, Vector3.up), Vector3.one);

		Matrix4x4 Root = Matrix4x4.TRS(user.Head.position, User_FixedRoot.GetRotation(), Vector3.one);
		Vector3 Gaze = user.Gaze.forward;
		Vector3 EF_RightHand = user.RightHand_IndexFingerTip.position - user.Head.position;
		Vector3 EF_LeftHand = user.LeftHand_IndexFingerTip.position - user.Head.position;

		float Default_Head = Quaternion.Angle(user.Head.rotation.GetRelativeRotationTo(User_FixedRoot), User_FixedRoot.GetRotation().GetRelativeRotationTo(User_FixedRoot));
		float Default_RightHand = Mathf.Max(user.RightHand.position.y, User_RightHand_Default.y) - User_RightHand_Default.y;
		float Default_LeftHand = Mathf.Max(user.LeftHand.position.y, User_LeftHand_Default.y) - User_LeftHand_Default.y;

		// Closest target (within 10 degrees) along the gaze / eye-finger rays.
		bool gaze_bool = false;
		float gaze_detected_angle = 180f;
		float ef_righthand_detected_angle = 180f;
		float ef_lefthand_detected_angle = 180f;
		for (int i = 0; i < Targets_GameObjects_User.Length; i++)
		{
			Vector3 ET = Targets_GameObjects_User[i].transform.position - Root.GetPosition();
			float angle = Vector3.Angle(ET.normalized, Gaze.normalized);
			if (angle < 10f)
			{
				if (gaze_detected_angle > angle)
				{
					gaze_detected = ET.normalized;
					gaze_detected_angle = angle;
				}
				gaze_bool = true;
			}
			angle = Vector3.Angle(ET.normalized, EF_RightHand.normalized);
			if (angle < 10f && ef_righthand_detected_angle > angle)
			{
				ef_righthand_detected = ET.normalized;
				ef_righthand_detected_angle = angle;
			}
			angle = Vector3.Angle(ET.normalized, EF_LeftHand.normalized);
			if (angle < 10f && ef_lefthand_detected_angle > angle)
			{
				ef_lefthand_detected = ET.normalized;
				ef_lefthand_detected_angle = angle;
			}
		}

		if (Default_Head < 10f)
		{
			gaze_corrected = gaze_bool ? Vector3.Lerp(Gaze, gaze_detected, Gaze_Progression) : Vector3.forward;
		}
		else
		{
			gaze_corrected = Vector3.Lerp(Gaze, gaze_detected, Gaze_Progression);
		}

		Vector3 Direction = gaze_corrected.normalized.GetRelativeDirectionTo(Root);
		input_float[0] = Direction.x;
		input_float[1] = Direction.y;
		input_float[2] = Direction.z;
		Direction = EF_RightHand.normalized.GetRelativeDirectionTo(Root);
		input_float[3] = Direction.x;
		input_float[4] = Direction.y;
		input_float[5] = Direction.z;
		Direction = EF_LeftHand.normalized.GetRelativeDirectionTo(Root);
		input_float[6] = Direction.x;
		input_float[7] = Direction.y;
		input_float[8] = Direction.z;
		// Idle flags (head is idle only when it is at rest and not looking at a target).
		input_float[9] = (Default_Head < 10f && !IsLookingAtAnyTarget(Gaze)) ? 1f : 0f;
		input_float[10] = Default_RightHand < 0.10f ? 1f : 0f;
		input_float[11] = Default_LeftHand < 0.10f ? 1f : 0f;

		return input_float;
	}

	public float[] CalculateInput_UserPose(int input_size)
	{
		float[] input_float = new float[input_size];
		Vector3 forward, up, pos;

		// Head relative to its target
		Matrix4x4 head_target_mat = Matrix4x4.TRS(User_Head_Target, Quaternion.LookRotation(User_FixedRoot.GetPosition() - User_Head_Target), Vector3.one);
		forward = user.Head.forward.GetRelativeDirectionTo(head_target_mat);
		up = user.Head.up.GetRelativeDirectionTo(head_target_mat);
		pos = (user.Head.position - User_FixedRoot.GetPosition()).magnitude * user.Head.position.GetRelativePositionTo(head_target_mat).normalized;
		Write(input_float, 0, forward, up, pos);

		// Hands relative to the head
		forward = user.Head.InverseTransformDirection(user.RightHand.forward);
		up = user.Head.InverseTransformDirection(user.RightHand.up);
		pos = (1f / User_Scale) * user.Head.InverseTransformPoint(user.RightHand.position);
		Write(input_float, 9, forward, up, pos);

		forward = user.Head.InverseTransformDirection(user.LeftHand.forward);
		up = user.Head.InverseTransformDirection(user.LeftHand.up);
		pos = (1f / User_Scale) * user.Head.InverseTransformPoint(user.LeftHand.position);
		Write(input_float, 18, forward, up, pos);

		return input_float;
	}

	public float[] CalculateInput_AvatarPose(int input_size)
	{
		float[] input_float = new float[input_size];
		Vector3 forward, up, pos;

		// Head
		forward = avatar.Head.forward.GetRelativeDirectionTo(Avatar_FixedRoot);
		up = avatar.Head.up.GetRelativeDirectionTo(Avatar_FixedRoot);
		pos = avatar.Head.position.GetRelativePositionTo(Avatar_FixedRoot);
		Write(input_float, 0, forward, up, pos);
		forward = avatar.Head.InverseTransformDirection(avatar.Gaze.forward);
		input_float[9] = forward.x;
		input_float[10] = forward.y;
		input_float[11] = forward.z;
		// Right hand
		forward = avatar.RightHand.forward.GetRelativeDirectionTo(Avatar_FixedRoot);
		up = avatar.RightHand.up.GetRelativeDirectionTo(Avatar_FixedRoot);
		pos = avatar.Head.TransformPoint((1f / Avatar_Scale) * avatar.Head.InverseTransformPoint(avatar.RightHand.position)).GetRelativePositionTo(Avatar_FixedRoot);
		Write(input_float, 12, forward, up, pos);
		pos = avatar.RightHand.InverseTransformPoint(avatar.RightHand_IndexFingerTip.position);
		input_float[21] = pos.x;
		input_float[22] = pos.y;
		input_float[23] = pos.z;
		// Left hand
		forward = avatar.LeftHand.forward.GetRelativeDirectionTo(Avatar_FixedRoot);
		up = avatar.LeftHand.up.GetRelativeDirectionTo(Avatar_FixedRoot);
		pos = avatar.Head.TransformPoint((1f / Avatar_Scale) * avatar.Head.InverseTransformPoint(avatar.LeftHand.position)).GetRelativePositionTo(Avatar_FixedRoot);
		Write(input_float, 24, forward, up, pos);
		pos = avatar.LeftHand.InverseTransformPoint(avatar.LeftHand_IndexFingerTip.position);
		input_float[33] = pos.x;
		input_float[34] = pos.y;
		input_float[35] = pos.z;

		return input_float;
	}

	public float[] CalculateInput_BP_GTNet(int input_size)
	{
		float[] input_float = new float[input_size];
		Vector3 Gaze = user.Gaze.forward;
		Vector3 EF_RightHand = user.RightHand_IndexFingerTip.position - user.Head.position;
		Vector3 EF_LeftHand = user.LeftHand_IndexFingerTip.position - user.Head.position;
		float Default_Head = Quaternion.Angle(user.Head.rotation.GetRelativeRotationTo(User_FixedRoot), User_FixedRoot.GetRotation().GetRelativeRotationTo(User_FixedRoot));
		float Default_RightHand = Mathf.Max(user.RightHand.position.y, User_RightHand_Default.y) - User_RightHand_Default.y;
		float Default_LeftHand = Mathf.Max(user.LeftHand.position.y, User_LeftHand_Default.y) - User_LeftHand_Default.y;

		// Estimated targets in the avatar's environment (spherical, relative to the avatar root)
		WriteSpherical(input_float, 0, Head_Target_Name, Avatar_Head_Target, Avatar_FixedRoot, editor.IsPoint);
		WriteSpherical(input_float, 3, RightHand_Target_Name, Avatar_RightHand_Target, Avatar_FixedRoot, editor.IsPoint);
		WriteSpherical(input_float, 6, LeftHand_Target_Name, Avatar_LeftHand_Target, Avatar_FixedRoot, editor.IsPoint);

		// Is the gaze / eye-finger ray on any target?
		input_float[9] = 0f;
		input_float[10] = 0f;
		input_float[11] = 0f;
		for (int i = 0; i < Targets_GameObjects_User.Length; i++)
		{
			Vector3 ET = Targets_GameObjects_User[i].transform.position - user.Head.position;
			if (Vector3.Angle(ET.normalized, Gaze.normalized) < 10f) input_float[9] = 1f;
			if (Vector3.Angle(ET.normalized, EF_RightHand.normalized) < 10f) input_float[10] = 1f;
			if (Vector3.Angle(ET.normalized, EF_LeftHand.normalized) < 10f) input_float[11] = 1f;
		}

		// Idle flags
		input_float[12] = (Default_Head < 10f && !IsLookingAtAnyTarget(Gaze)) ? 1f : 0f;
		input_float[13] = Default_RightHand < 0.10f ? 1f : 0f;
		input_float[14] = Default_LeftHand < 0.10f ? 1f : 0f;

		// Confidence of the current target assignment
		input_float[15] = Confidence_Head;
		input_float[16] = Confidence_RightHand;
		input_float[17] = Confidence_LeftHand;

		// Head target in the user's environment (distance is always 1 here)
		WriteSpherical(input_float, 18, Head_Target_Name, User_Head_Target, User_FixedRoot, true);

		return input_float;
	}

	private static void Write(float[] buffer, int offset, Vector3 forward, Vector3 up, Vector3 pos)
	{
		buffer[offset + 0] = forward.x;
		buffer[offset + 1] = forward.y;
		buffer[offset + 2] = forward.z;
		buffer[offset + 3] = up.x;
		buffer[offset + 4] = up.y;
		buffer[offset + 5] = up.z;
		buffer[offset + 6] = pos.x;
		buffer[offset + 7] = pos.y;
		buffer[offset + 8] = pos.z;
	}

	private static void WriteSpherical(float[] buffer, int offset, string targetName, Vector3 target, Matrix4x4 root, bool unitDistance)
	{
		if (targetName == "TargetNull")
		{
			buffer[offset + 0] = 0f;
			buffer[offset + 1] = 0f;
			buffer[offset + 2] = 0f;
			return;
		}
		Vector3 direction = (target - root.GetPosition()).normalized.GetRelativeDirectionTo(root);
		buffer[offset + 0] = Mathf.Atan2(direction.x, direction.z);
		buffer[offset + 1] = Mathf.Atan2(direction.y, Mathf.Sqrt(direction.x * direction.x + direction.z * direction.z));
		buffer[offset + 2] = unitDistance ? 1f : Mathf.Clamp((target - root.GetPosition()).magnitude, 0f, 1f);
	}

	public string SetInput(string which, int input_size)
	{
		float[] sequence;
		switch (which)
		{
			case "TENet": sequence = CalculateInput_TENet(input_size); break;
			case "BP_GTNet": sequence = CalculateInput_BP_GTNet(input_size); break;
			case "UserPose": sequence = CalculateInput_UserPose(input_size); break;
			case "AvatarPose": sequence = CalculateInput_AvatarPose(input_size); break;
			default: throw new ArgumentException("Unknown input " + which);
		}
		return string.Join(", ", sequence.Select(f => f.ToString("F8")));
	}

	/* ---------------------------------------------------------------- Network output */

	public void SetOutput(string[] output)
	{
		if (output.Length < 44)
		{
			Debug.LogWarning("Unexpected reply from the inference server (" + output.Length + " values).");
			return;
		}
		Vector3 forward, up, pos;

		// Head + gaze
		forward = ParseVector3(output, 0).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		up = ParseVector3(output, 3).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		pos = ParseVector3(output, 6).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.Head.rotation = Quaternion.LookRotation(forward, up);
		avatar.Head.position = pos;
		avatar.Gaze.position = avatar.Head.position;
		avatar.Gaze.forward = avatar.Head.TransformDirection(ParseVector3(output, 9).normalized);

		// Right hand (+ finger tip in hand space)
		forward = ParseVector3(output, 12).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		up = ParseVector3(output, 15).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		pos = ParseVector3(output, 18).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.RightHand.rotation = Quaternion.LookRotation(forward, up);
		avatar.RightHand.position = avatar.Head.TransformPoint(Avatar_Scale * avatar.Head.InverseTransformPoint(pos));
		avatar.RightHand_IndexFingerTip.position = avatar.RightHand.TransformPoint(Avatar_Scale * ParseVector3(output, 21));

		// Left hand (+ finger tip in hand space)
		forward = ParseVector3(output, 24).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		up = ParseVector3(output, 27).normalized.GetRelativeDirectionFrom(Avatar_FixedRoot);
		pos = ParseVector3(output, 30).GetRelativePositionFrom(Avatar_FixedRoot);
		avatar.LeftHand.rotation = Quaternion.LookRotation(forward, up);
		avatar.LeftHand.position = avatar.Head.TransformPoint(Avatar_Scale * avatar.Head.InverseTransformPoint(pos));
		avatar.LeftHand_IndexFingerTip.position = avatar.LeftHand.TransformPoint(Avatar_Scale * ParseVector3(output, 33));

		// Progressions
		Gaze_Progression = Parse(output[36]);
		DeicticGesture_Progression_RightHand = Parse(output[37]);
		DeicticGesture_Progression_LeftHand = Parse(output[38]);
		Idle_Progression_Head = Parse(output[39]);
		Idle_Progression_RightHand = Parse(output[40]);
		Idle_Progression_LeftHand = Parse(output[41]);

		// Targets: estimated by TENet, or the ground-truth labels of the clip
		TargetModule module = SetGTTarget ? (TargetModule)editor.GetData().GetModule(Module.ID.Target) : null;
		if (module != null)
		{
			int frame = editor.GetCurrentFrame().Index - 1;
			Confidence = 1f;
			Head_Target_Name = module.Head_Target.Props[frame];
			RightHand_Target_Name = !editor.Mirroring ? module.RightHand_Target.Props[frame] : module.LeftHand_Target.Props[frame];
			LeftHand_Target_Name = !editor.Mirroring ? module.LeftHand_Target.Props[frame] : module.RightHand_Target.Props[frame];
		}
		else
		{
			Confidence = Parse(output[42]);
			int[] selected = DecodeIndexToTargetIndices((int)Parse(output[43]));
			Head_Target_Name = Targets_User[selected[0]].id;
			RightHand_Target_Name = Targets_User[selected[1]].id;
			LeftHand_Target_Name = Targets_User[selected[2]].id;
		}

		UpdateConfidence();

		Head_Target_Name_previous = Head_Target_Name;
		RightHand_Target_Name_previous = RightHand_Target_Name;
		LeftHand_Target_Name_previous = LeftHand_Target_Name;
	}

	// Confidence grows while the user keeps gazing at the estimated head target and resets when the estimate changes.
	private void UpdateConfidence()
	{
		bool gazeOnHeadTarget = false;
		if (Head_Target_Name != "TargetNull")
		{
			Vector3 ET = userTargets[Head_Target_Name].position - user.Head.position;
			gazeOnHeadTarget = Vector3.Angle(ET.normalized, user.Gaze.forward) < 10f;
		}

		if (Head_Target_Name == Head_Target_Name_previous)
		{
			if (Head_Target_Name != "TargetNull")
			{
				Confidence_Head = Step(Confidence_Head, gazeOnHeadTarget);
			}
		}
		else
		{
			Confidence_Head = 0f;
		}

		if (RightHand_Target_Name == RightHand_Target_Name_previous)
		{
			Confidence_RightHand = Step(Confidence_RightHand, RightHand_Target_Name != "TargetNull" && gazeOnHeadTarget);
		}
		else
		{
			Confidence_RightHand = 0f;
		}

		if (LeftHand_Target_Name == LeftHand_Target_Name_previous)
		{
			Confidence_LeftHand = Step(Confidence_LeftHand, LeftHand_Target_Name != "TargetNull" && gazeOnHeadTarget);
		}
		else
		{
			Confidence_LeftHand = 0f;
		}
	}

	private static float Step(float value, bool increase)
	{
		return Mathf.Clamp01(value + (increase ? 1f : -1f) / 15f);
	}

	private static float Parse(string value)
	{
		return float.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
	}

	private static Vector3 ParseVector3(string[] values, int offset)
	{
		return new Vector3(Parse(values[offset]), Parse(values[offset + 1]), Parse(values[offset + 2]));
	}

	/* ---------------------------------------------------------------- Drawing */

	private void DrawTargetShape(int index, Vector3 position, float size)
	{
		Quaternion rotation = Quaternion.Euler(0f, 100f * Time.time, 0f);
		switch (index % 3)
		{
			case 0: UltiDraw.DrawWiredSphere(position, rotation, size, UltiDraw.White, UltiDraw.White); break;
			case 1: UltiDraw.DrawWiredPyramid(position - new Vector3(0f, size / 2f, 0f), rotation, size, size, UltiDraw.White, UltiDraw.White); break;
			default: UltiDraw.DrawWiredCube(position, rotation, size, UltiDraw.White, UltiDraw.White); break;
		}
	}

	private void DrawEstimatedTargets(bool withConfidence)
	{
		bool point = editor.IsPoint;
		float lift = point ? 0.025f : 0.0125f;
		float size = point ? 0.12f : 0.06f;
		if (Head_Target_Name != "TargetNull")
		{
			float s = withConfidence ? size + (1f - Confidence_Head) * (point ? 1f : 0.5f) : size;
			Color c = withConfidence ? new Color(1f, 0f, 0f, 0.05f + Mathf.Pow(Confidence_Head, 2f)) : Color.red;
			UltiDraw.DrawSphere(User_Head_Target + new Vector3(0f, lift, 0f), Quaternion.identity, s, c);
			UltiDraw.DrawSphere(Avatar_Head_Target + new Vector3(0f, lift, 0f), Quaternion.identity, s, c);
		}
		if (RightHand_Target_Name != "TargetNull")
		{
			float s = withConfidence ? size + (1f - Confidence_RightHand) * (point ? 1f : 0.5f) : size;
			Color c = withConfidence ? new Color(0f, 1f, 0f, 0.05f + Mathf.Pow(Confidence_RightHand, 2f)) : Color.green;
			UltiDraw.DrawSphere(User_RightHand_Target, Quaternion.identity, s, c);
			UltiDraw.DrawSphere(Avatar_RightHand_Target, Quaternion.identity, s, c);
		}
		if (LeftHand_Target_Name != "TargetNull")
		{
			float s = withConfidence ? size + (1f - Confidence_LeftHand) * (point ? 1f : 0.5f) : size;
			Color c = withConfidence ? new Color(0f, 0f, 1f, 0.05f + Mathf.Pow(Confidence_LeftHand, 2f)) : Color.blue;
			UltiDraw.DrawSphere(User_LeftHand_Target - new Vector3(0f, lift, 0f), Quaternion.identity, s, c);
			UltiDraw.DrawSphere(Avatar_LeftHand_Target - new Vector3(0f, lift, 0f), Quaternion.identity, s, c);
		}
	}

	private static void DrawEndEffectors(Points p)
	{
		// Head
		UltiDraw.DrawTranslateGizmo(p.Head.position, p.Head.rotation, 0.15f);
		UltiDraw.DrawSphere(p.Head.position, p.Head.rotation, 0.15f, new Color(1f, 0f, 0f, 0.5f));
		UltiDraw.DrawLine(p.Head.position, p.Head.position + 0.5f * p.Gaze.forward, 0.01f, 0.0f, UltiDraw.Red);
		// Right hand
		UltiDraw.DrawTranslateGizmo(p.RightHand.position, p.RightHand.rotation, 0.075f);
		UltiDraw.DrawSphere(p.RightHand.position, p.RightHand.rotation, 0.10f, new Color(0f, 1f, 0f, 0.5f));
		UltiDraw.DrawSphere(p.RightHand_IndexFingerTip.position, Quaternion.identity, 0.05f, new Color(0f, 1f, 0f, 0.5f));
		// Left hand
		UltiDraw.DrawTranslateGizmo(p.LeftHand.position, p.LeftHand.rotation, 0.075f);
		UltiDraw.DrawSphere(p.LeftHand.position, p.LeftHand.rotation, 0.10f, new Color(0f, 0f, 1f, 0.5f));
		UltiDraw.DrawSphere(p.LeftHand_IndexFingerTip.position, Quaternion.identity, 0.05f, new Color(0f, 0f, 1f, 0.5f));
	}

	void OnRenderObject()
	{
		if (user == null || avatar == null || editor == null || editor.GetData() == null)
		{
			return;
		}

		UltiDraw.Begin();

		if (DrawTarget)
		{
			float size = editor.IsPoint ? 0.15f : 0.075f;
			for (int i = 0; i < Targets_GameObjects_User.Length; i++)
			{
				DrawTargetShape(i, Targets_GameObjects_User[i].transform.position, size);
			}
			for (int i = 0; i < Targets_GameObjects_Avatar.Length; i++)
			{
				DrawTargetShape(i, Targets_GameObjects_Avatar[i].transform.position, size);
			}
		}

		if (DrawEstimatedTarget)
		{
			DrawEstimatedTargets(false);
		}
		if (DrawEstimatedTarget_Confidence)
		{
			DrawEstimatedTargets(true);
		}

		if (DrawUserEE)
		{
			DrawEndEffectors(user);
		}
		if (DrawAvatarEE)
		{
			DrawEndEffectors(avatar);
		}

		if (DrawRay && editor.IsPoint)
		{
			Vector3 Gaze_User = user.Gaze.forward;
			Vector3 EF_RightHand_User = user.RightHand_IndexFingerTip.position - user.Head.position;
			Vector3 EF_LeftHand_User = user.LeftHand_IndexFingerTip.position - user.Head.position;
			Vector3 Gaze_Avatar = avatar.Gaze.forward;
			Vector3 EF_RightHand_Avatar = avatar.RightHand_IndexFingerTip.position - avatar.Head.position;
			Vector3 EF_LeftHand_Avatar = avatar.LeftHand_IndexFingerTip.position - avatar.Head.position;

			if (Head_Target_Name != "TargetNull" && Vector3.Angle((User_Head_Target - user.Head.position).normalized, Gaze_User.normalized) < 10f)
			{
				UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * Gaze_User, 0.005f, 0.015f, UltiDraw.Red);
				UltiDraw.DrawLine(avatar.Head.position, avatar.Head.position + 5f * Gaze_Avatar, 0.005f, 0.015f, UltiDraw.Red);
			}
			if (RightHand_Target_Name != "TargetNull" && Vector3.Angle((User_RightHand_Target - user.Head.position).normalized, EF_RightHand_User.normalized) < 10f)
			{
				UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * EF_RightHand_User, 0.005f, 0.015f, UltiDraw.Green);
				UltiDraw.DrawLine(avatar.Head.position, avatar.Head.position + 5f * EF_RightHand_Avatar, 0.005f, 0.015f, UltiDraw.Green);
			}
			if (LeftHand_Target_Name != "TargetNull" && Vector3.Angle((User_LeftHand_Target - user.Head.position).normalized, EF_LeftHand_User.normalized) < 10f)
			{
				UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * EF_LeftHand_User, 0.005f, 0.015f, UltiDraw.Blue);
				UltiDraw.DrawLine(avatar.Head.position, avatar.Head.position + 5f * EF_LeftHand_Avatar, 0.005f, 0.015f, UltiDraw.Blue);
			}
		}

		if (DrawRayTENet)
		{
			Vector3 Gaze = user.Gaze.forward;
			Vector3 EF_RightHand = user.RightHand_IndexFingerTip.position - user.Head.position;
			Vector3 EF_LeftHand = user.LeftHand_IndexFingerTip.position - user.Head.position;
			UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * Gaze, new Color(1f, 0f, 0f, 0.25f));
			UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * EF_RightHand, new Color(0f, 1f, 0f, 0.25f));
			UltiDraw.DrawLine(user.Head.position, user.Head.position + 5f * EF_LeftHand, new Color(0f, 0f, 1f, 0.25f));
		}

		if (DrawRoot)
		{
			UltiDraw.DrawTranslateGizmo(User_FixedRoot.GetPosition(), User_FixedRoot.GetRotation(), 0.25f);
			UltiDraw.DrawTranslateGizmo(Avatar_FixedRoot.GetPosition(), Avatar_FixedRoot.GetRotation(), 0.25f);
		}

		UltiDraw.End();
	}

	// Normalized GUI coordinates (origin top-left) of the 4-way split:
	// left column x = 0.01, right column x = 0.505; Up row top = 0.01, Front row top = 0.505; bottom edge = 0.99.
	private const float CaptionLeft = 0.02f, CaptionRight = 0.515f, CaptionTop = 0.02f, CaptionBottom = 0.515f;
	private const float FooterY = 0.935f;

	void OnGUI()
	{
		UltiDraw.Begin();
		if (DrawCaptions)
		{
			UltiDraw.DrawGUILabel(CaptionLeft, CaptionTop, 0.016f, "User (Up)", UltiDraw.White);
			UltiDraw.DrawGUILabel(CaptionRight, CaptionTop, 0.016f, "Avatar (Up)", UltiDraw.White);
			UltiDraw.DrawGUILabel(CaptionLeft, CaptionBottom, 0.016f, "User (Front)", UltiDraw.White);
			UltiDraw.DrawGUILabel(CaptionRight, CaptionBottom, 0.016f, "Avatar (Front)", UltiDraw.White);
		}
		// Status / timing at the bottom-left of the screen (inside the User (Front) viewport)
		if (editor == null || editor.GetData() == null)
		{
			UltiDraw.DrawGUILabel(CaptionLeft, FooterY + 0.025f, 0.014f, "No motion data loaded. Select a Split / Subject / Category on the MotionEditor and press Import.", UltiDraw.White);
		}
		else if (DrawFPS)
		{
			UltiDraw.DrawGUILabel(CaptionLeft, FooterY, 0.014f, "Inference: " + inferenceTime.ToString("00.00") + " ms", UltiDraw.White);
			UltiDraw.DrawGUILabel(CaptionLeft, FooterY + 0.025f, 0.014f, "FPS: " + inferenceFPS.ToString("00.0"), UltiDraw.White);
		}
		UltiDraw.End();
	}
}
#endif
