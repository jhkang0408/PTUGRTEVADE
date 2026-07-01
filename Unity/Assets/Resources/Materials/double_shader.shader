Shader "Unlit/double_shader"
{
    Properties
    {
        _Color("Main Color", Color)=(1,1,1,1)
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        cull off
        Pass
        {
            Lighting Off

            SetTexture[_MainTex]
            {
            constantColor[_Color]
            combine constant * texture
            }
        }
    }
}
