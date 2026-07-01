#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

public class MotionExporter_BP_GTNet : MotionExporter
{
	protected override string NetworkName => "BP_GTNet";

	[MenuItem("Motion Data Processing/Motion Exporter (BP_GTNet)")]
	static void Init()
	{
		Window = EditorWindow.GetWindow(typeof(MotionExporter_BP_GTNet));
		Scroll = Vector3.zero;
	}

	protected override void WriteFrame(Data X, Sample current, Sample initial)
    {
		Vector3 Direction;
		float HorizontalAngle, VerticalAngle, Distance;
		Vector3 Head_Target = GameObject.Find(current.Head_Target_name).transform.position;
		Vector3 RightHand_Target = GameObject.Find(current.RightHand_Target_name).transform.position;
		Vector3 LeftHand_Target = GameObject.Find(current.LeftHand_Target_name).transform.position;
		Vector3 ET_Head = Head_Target - initial.Root.GetPosition();
		Vector3 ET_RightHand = RightHand_Target - initial.Root.GetPosition();
		Vector3 ET_LeftHand = LeftHand_Target - initial.Root.GetPosition();
		
		/* Target */
		// Head Target
		Direction = ET_Head.normalized.GetRelativeDirectionTo(initial.Root);
		if (current.Head_Target_name=="TargetNull")
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
		X.Feed(HorizontalAngle, "Head_Target_Horizontal");
		X.Feed(VerticalAngle, "Head_Target_Vertical");
		X.Feed(Distance, "Head_Target_Distance");
		// RightHand Target
		Direction = ET_RightHand.normalized.GetRelativeDirectionTo(initial.Root);
		if (current.RightHand_Target_name=="TargetNull")
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
		X.Feed(HorizontalAngle, "RightHand_Target_Horizontal");
		X.Feed(VerticalAngle, "RightHand_Target_Vertical");
		X.Feed(Distance, "RightHand_Target_Distance");
		// LeftHand Target
		Direction = ET_LeftHand.normalized.GetRelativeDirectionTo(initial.Root);
		if (current.LeftHand_Target_name=="TargetNull")
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
		X.Feed(HorizontalAngle, "LeftHand_Target_Horizontal");
		X.Feed(VerticalAngle, "LeftHand_Target_Vertical");
		X.Feed(Distance, "LeftHand_Target_Distance");

		Vector3 Gaze = current.Gaze;		
		Vector3 EF_RightHand = current.RightHand_IndexFingerTip - current.Head.GetPosition();
		Vector3 EF_LeftHand = current.LeftHand_IndexFingerTip - current.Head.GetPosition();
		float Default_Head = Quaternion.Angle(current.Head.GetRotation(), initial.Root.GetRotation());
		float Default_RightHand = Mathf.Max(current.RightHand.GetPosition().y, initial.RightHand.GetPosition().y) - initial.RightHand.GetPosition().y;
		float Default_LeftHand = Mathf.Max(current.LeftHand.GetPosition().y, initial.LeftHand.GetPosition().y) - initial.LeftHand.GetPosition().y;

		// Ray Feature
		int Ray_Head = 0;
		int Ray_RightHand = 0;
		int Ray_LeftHand = 0;
		for (int i = 0; i < Editor.Targets.Length; i++)
		{
			Vector3 ET = Editor.Targets[i].transform.position - current.Head.GetPosition();
			if (Editor.Targets[i].transform.position.y > 0)
			{
				if (Vector3.Angle(ET.normalized, Gaze.normalized) < 10f)
				{
					Ray_Head = 1;
				}
				if (Vector3.Angle(ET.normalized, EF_RightHand.normalized) < 10f)
				{
					Ray_RightHand = 1;
				}
				if (Vector3.Angle(ET.normalized, EF_LeftHand.normalized) < 10f)
				{
					Ray_LeftHand = 1;
				}
			}
		}
		X.Feed(Ray_Head, "Ray_Head");
		X.Feed(Ray_RightHand, "Ray_RightHand");
		X.Feed(Ray_LeftHand, "Ray_LeftHand");

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

		/* End-Effector (Encoder) */
		// Head
		Matrix4x4 Head_Target_mat = Matrix4x4.TRS(Head_Target, Quaternion.LookRotation(initial.Root.GetPosition() - Head_Target), Vector3.one);
		X.Feed(current.Head.GetForward().GetRelativeDirectionTo(Head_Target_mat), "Head Forward (Encoder)");
		X.Feed(current.Head.GetUp().GetRelativeDirectionTo(Head_Target_mat), "Head Up (Encoder)");
		X.Feed((current.Head.GetPosition() - initial.Root.GetPosition()).magnitude * (current.Head.GetPosition().GetRelativePositionTo(Head_Target_mat).normalized), "Head Position (Encoder)");

		// RightHand
		X.Feed(current.RightHand.GetForward().GetRelativeDirectionTo(current.Head), "RightHand Forward (Encoder)");
		X.Feed(current.RightHand.GetUp().GetRelativeDirectionTo(current.Head), "RightHand Up (Encoder)");
		X.Feed(current.RightHand.GetPosition().GetRelativePositionTo(current.Head), "RightHand Position (Encoder)");

		// LeftHand
		X.Feed(current.LeftHand.GetForward().GetRelativeDirectionTo(current.Head), "LeftHand Forward (Encoder)");
		X.Feed(current.LeftHand.GetUp().GetRelativeDirectionTo(current.Head), "LeftHand Up (Encoder)");
		X.Feed(current.LeftHand.GetPosition().GetRelativePositionTo(current.Head), "LeftHand Position (Encoder)");

		/* End-Effector (Decoder) */
		// Head		
		X.Feed(current.Head.GetForward().GetRelativeDirectionTo(initial.Root), "Head Forward (Decoder)");
		X.Feed(current.Head.GetUp().GetRelativeDirectionTo(initial.Root), "Head Up (Decoder)");
		X.Feed(current.Head.GetPosition().GetRelativePositionTo(initial.Root), "Head Position (Decoder)");
		X.Feed(current.Gaze.GetRelativeDirectionTo(current.Head), "Gaze Forward (Decoder)");

		// RightHand
		X.Feed(current.RightHand.GetForward().GetRelativeDirectionTo(initial.Root), "RightHand Forward (Decoder)");
		X.Feed(current.RightHand.GetUp().GetRelativeDirectionTo(initial.Root), "RightHand Up (Decoder)");
		X.Feed(current.RightHand.GetPosition().GetRelativePositionTo(initial.Root), "RightHand Position (Decoder)");
		X.Feed(current.RightHand_IndexFingerTip.GetRelativePositionTo(current.RightHand), "RightHand_IndexFingerTip Position (Decoder)");

		// LeftHand
		X.Feed(current.LeftHand.GetForward().GetRelativeDirectionTo(initial.Root), "LeftHand Forward (Decoder)");
		X.Feed(current.LeftHand.GetUp().GetRelativeDirectionTo(initial.Root), "LeftHand Up (Decoder)");
		X.Feed(current.LeftHand.GetPosition().GetRelativePositionTo(initial.Root), "LeftHand Position (Decoder)");
		X.Feed(current.LeftHand_IndexFingerTip.GetRelativePositionTo(current.LeftHand), "LeftHand_IndexFingerTip Position (Decoder)");

		// Write Line
		X.Store();
	}
}
#endif
