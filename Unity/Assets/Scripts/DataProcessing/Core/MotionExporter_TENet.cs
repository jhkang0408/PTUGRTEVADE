#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class MotionExporter_TENet : MotionExporter
{
	protected override string NetworkName => "TENet";

	[MenuItem("Motion Data Processing/Motion Exporter (TENet)")]
	static void Init()
	{
		Window = EditorWindow.GetWindow(typeof(MotionExporter_TENet));
		Scroll = Vector3.zero;
	}

	protected override void WriteFrame(Data X, Sample current, Sample initial)
	{
		Vector3 Direction;
		float HorizontalAngle, VerticalAngle, Distance;
		Matrix4x4 Root = Matrix4x4.TRS(current.Head.GetPosition(), initial.Root.GetRotation(), Vector3.one);
		Vector3 Head_Target = GameObject.Find(current.Head_Target_name).transform.position;
		Vector3 RightHand_Target = GameObject.Find(current.RightHand_Target_name).transform.position;
		Vector3 LeftHand_Target = GameObject.Find(current.LeftHand_Target_name).transform.position;
		Vector3 ET_Head = Head_Target - current.Head.GetPosition();
		Vector3 ET_RightHand = RightHand_Target - current.Head.GetPosition();
		Vector3 ET_LeftHand = LeftHand_Target - current.Head.GetPosition();
		Vector3 Gaze = current.Gaze;
		Vector3 EF_RightHand = current.RightHand_IndexFingerTip - current.Head.GetPosition();
		Vector3 EF_LeftHand = current.LeftHand_IndexFingerTip - current.Head.GetPosition();
		
		float Default_Head = Quaternion.Angle(current.Head.GetRotation(), initial.Root.GetRotation());
		float Default_RightHand = Mathf.Max(current.RightHand.GetPosition().y, initial.RightHand.GetPosition().y) - initial.RightHand.GetPosition().y;
		float Default_LeftHand = Mathf.Max(current.LeftHand.GetPosition().y, initial.LeftHand.GetPosition().y) - initial.LeftHand.GetPosition().y;

		// Gaze
		X.Feed(Gaze.GetRelativeDirectionTo(Root), "Gaze");
		// EF_RightHand
		X.Feed(EF_RightHand.normalized.GetRelativeDirectionTo(Root), "EF_RightHand");
		// EF_LeftHand
		X.Feed(EF_LeftHand.normalized.GetRelativeDirectionTo(Root), "EF_LeftHand");

		// Default Feature (Head)
		int Default = 1;
		if (Default_Head < 10f)
		{
			for (int i = 0; i < Editor.Targets.Length; i++)
			{
				Vector3 ET = Editor.Targets[i].transform.position - current.Head.GetPosition();
				if (Editor.Targets[i].transform.position.y > 0)
                {
					if (Vector3.Angle(ET.normalized, Gaze.normalized) < 10f)
					{
						Default = 0;
						break;
					}
				}
			}
		}
		else
		{
			Default = 0;
		}
		X.Feed(Default, "Default_Head");
		// Default Feature (RightHand)
		if (Default_RightHand < 0.10f)
		{
			X.Feed(1, "Default_RightHand");
		}
		else
		{
			X.Feed(0, "Default_RightHand");
		}
		// Default Feature (LeftHand)
		if (Default_LeftHand < 0.10f)
		{
			X.Feed(1, "Default_LeftHand");
		}
		else
		{
			X.Feed(0, "Default_LeftHand");
		}

		/* Target (TENet) */
		// Head
		Direction = ET_Head.normalized.GetRelativeDirectionTo(Root);
		if (current.Head_Target_name == "TargetNull")
		{
			HorizontalAngle = 0;
			VerticalAngle = 0;
			Distance = 0;
		}
		else
		{
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_Head.magnitude, 0f, 1f);
		}
		X.Feed(HorizontalAngle, "Head_Target_Horizontal (TENet)");
		X.Feed(VerticalAngle, "Head_Target_Vertical (TENet)");
		X.Feed(Distance, "Head_Target_Distance (TENet)");
		// RightHand 
		Direction = ET_RightHand.normalized.GetRelativeDirectionTo(Root);
		if (current.RightHand_Target_name == "TargetNull")
		{
			HorizontalAngle = 0;
			VerticalAngle = 0;
			Distance = 0;
		}
		else
		{
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_RightHand.magnitude, 0f, 1f);
		}
		X.Feed(HorizontalAngle, "RightHand_Target_Horizontal (TENet)");
		X.Feed(VerticalAngle, "RightHand_Target_Vertical (TENet)");
		X.Feed(Distance, "RightHand_Target_Distance (TENet)");
		// LeftHand
		Direction = ET_LeftHand.normalized.GetRelativeDirectionTo(Root);
		if (current.LeftHand_Target_name == "TargetNull")
		{
			HorizontalAngle = 0;
			VerticalAngle = 0;
			Distance = 0;
		}
		else
		{
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_LeftHand.magnitude, 0f, 1f);
		}
		X.Feed(HorizontalAngle, "LeftHand_Target_Horizontal (TENet)");
		X.Feed(VerticalAngle, "LeftHand_Target_Vertical (TENet)");
		X.Feed(Distance, "LeftHand_Target_Distance (TENet)");

		if (!IsTrain)
		{
			/* Targets (TENet) */
			// Target 1
			Direction = (Editor.Targets[0].position - Root.GetPosition()).normalized.GetRelativeDirectionTo(Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[0].position - Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target1_Horizontal (TENet)");
			X.Feed(VerticalAngle, "Target1_Vertical (TENet)");
			X.Feed(Distance, "Target1_Distance (TENet)");
			// Target 2
			Direction = (Editor.Targets[1].position - Root.GetPosition()).normalized.GetRelativeDirectionTo(Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[1].position - Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target2_Horizontal (TENet)");
			X.Feed(VerticalAngle, "Target2_Vertical (TENet)");
			X.Feed(Distance, "Target2_Distance (TENet)");
			// Target 3
			Direction = (Editor.Targets[2].position - Root.GetPosition()).normalized.GetRelativeDirectionTo(Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[2].position - Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target3_Horizontal (TENet)");
			X.Feed(VerticalAngle, "Target3_Vertical (TENet)");
			X.Feed(Distance, "Target3_Distance (TENet)");

			/* Target (UMNet) */
			ET_Head = Head_Target - initial.Root.GetPosition();
			ET_RightHand = RightHand_Target - initial.Root.GetPosition();
			ET_LeftHand = LeftHand_Target - initial.Root.GetPosition();
			// Head Target
			Direction = ET_Head.normalized.GetRelativeDirectionTo(initial.Root);
			if (current.Head_Target_name == "TargetNull")
			{
				HorizontalAngle = 0;
				VerticalAngle = 0;
				Distance = 0;
			}
			else
			{
				HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
				VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
				Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_Head.magnitude, 0f, 1f);
			}
			X.Feed(HorizontalAngle, "Head_Target_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "Head_Target_Vertical (UMNet)");
			X.Feed(Distance, "Head_Target_Distance (UMNet)");
			// RightHand Target
			Direction = ET_RightHand.normalized.GetRelativeDirectionTo(initial.Root);
			if (current.RightHand_Target_name == "TargetNull")
			{
				HorizontalAngle = 0;
				VerticalAngle = 0;
				Distance = 0;
			}
			else
			{
				HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
				VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
				Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_RightHand.magnitude, 0f, 1f);
			}
			X.Feed(HorizontalAngle, "RightHand_Target_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "RightHand_Target_Vertical (UMNet)");
			X.Feed(Distance, "RightHand_Target_Distance (UMNet)");
			// LeftHand Target
			Direction = ET_LeftHand.normalized.GetRelativeDirectionTo(initial.Root);
			if (current.LeftHand_Target_name == "TargetNull")
			{
				HorizontalAngle = 0;
				VerticalAngle = 0;
				Distance = 0;
			}
			else
			{
				HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
				VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
				Distance = Editor.IsPoint ? 1f : Mathf.Clamp(ET_LeftHand.magnitude, 0f, 1f);
			}
			X.Feed(HorizontalAngle, "LeftHand_Target_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "LeftHand_Target_Vertical (UMNet)");
			X.Feed(Distance, "LeftHand_Target_Distance (UMNet)"); 

			/* Targets (UMNet) */
			// Target 1
			Direction = (Editor.Targets[0].position - initial.Root.GetPosition()).normalized.GetRelativeDirectionTo(initial.Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[0].position - initial.Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target1_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "Target1_Vertical (UMNet)");
			X.Feed(Distance, "Target1_Distance (UMNet)");
			// Target 2
			Direction = (Editor.Targets[1].position - initial.Root.GetPosition()).normalized.GetRelativeDirectionTo(initial.Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[1].position - initial.Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target2_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "Target2_Vertical (UMNet)");
			X.Feed(Distance, "Target2_Distance (UMNet)");
			// Target 3
			Direction = (Editor.Targets[2].position - initial.Root.GetPosition()).normalized.GetRelativeDirectionTo(initial.Root);
			HorizontalAngle = Mathf.Atan2(Direction.x, Direction.z);
			VerticalAngle = Mathf.Atan2(Direction.y, Mathf.Sqrt(Direction.x * Direction.x + Direction.z * Direction.z));
			Distance = Editor.IsPoint ? 1f : Mathf.Clamp((Editor.Targets[2].position - initial.Root.GetPosition()).magnitude, 0f, 1f);
			X.Feed(HorizontalAngle, "Target3_Horizontal (UMNet)");
			X.Feed(VerticalAngle, "Target3_Vertical (UMNet)");
			X.Feed(Distance, "Target3_Distance (UMNet)");

			/* Head Position (Decoder) */
			X.Feed(current.Head.GetPosition().GetRelativePositionTo(initial.Root), "Head Position (Decoder)");
			// Gaze
			X.Feed(Gaze.GetRelativeDirectionTo(initial.Root), "Gaze (Fixed Root)");
			// EF_RightHand
			X.Feed(EF_RightHand.normalized.GetRelativeDirectionTo(initial.Root), "EF_RightHand (Fixed Root)");
			// EF_LeftHand
			X.Feed(EF_LeftHand.normalized.GetRelativeDirectionTo(initial.Root), "EF_LeftHand (Fixed Root)");
		}

		// Write Line
		X.Store();
	}
}
#endif
