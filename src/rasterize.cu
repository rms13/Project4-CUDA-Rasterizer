/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <thrust/execution_policy.h>
#include <thrust/partition.h>

#include <chrono>

#define POINTS 0
#define WIREFRAME 0

#define DEBUG_DEPTH 0
#define DEBUG_NORMALS 0

#define TEXTURE 0
#define TEXTURE_PERSP_CORRECT 0
#define TEXTURE_BILINEAR_FILT 0

#define BLUR 0
#define BLUR_SHARED 0

#define SSAO 0 // SCREEN SPACE AMBIENT OCCLUSION : WORK UNDER PROGRESS 

#define BBOX_OPTIMIZATIONS 1
#define BACK_FACE_CULLING 0
#define BACK_FACE_CULLING_WITHOUT_COMPACTION 0

#define SSAA 2 // SUPERSAMPLE ANTIALIASING
               // 1 (min value : no AA), 2, 4, ...

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType {
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
    glm::vec4 pos;

    // TODO: add new attributes to your VertexOut
    // The attributes listed below might be useful, 
    // but always feel free to modify on your own

    glm::vec3 eyePos;	// eye space position used for shading
    glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
    //glm::vec3 col;
    glm::vec2 texcoord0;
    TextureData* dev_diffuseTex = NULL;
    int texWidth, texHeight;
    // ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
    bool back = false; // used for back face culling
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex = NULL;
    int texWidth, texHeight;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;

static int width = 0;
static int height = 0;

static int totalNumPrimitivesCompact = 0; // updated after compaction by culling
static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
static int * dev_mutex = NULL;

static glm::vec3 *dev_postBuffer1 = NULL;
static glm::vec3 *dev_postBuffer2 = NULL;

static Primitive *dev_primitives_compact = NULL;

// gaussian kernel
// source: http://www.sunsetlakesoftware.com/2013/10/21/optimizing-gaussian-blurs-mobile-gpu
// source: http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
__constant__ float mat[5] =
  { 0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162 };

__constant__ float light[3] = {100.f, 100.f, 100.f};

/**
* Called once at the beginning of the program to allocate memory.
*/
void rasterizeInit(int w, int h) {
  width = w * SSAA;
  height = h * SSAA;
  cudaFree(dev_fragmentBuffer);
  cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  cudaFree(dev_framebuffer);
  cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
  cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

  cudaMalloc(&dev_postBuffer1, width * height * sizeof(glm::vec3));
  cudaMemset(dev_postBuffer1, 0, width * height * sizeof(glm::vec3));
  cudaMalloc(&dev_postBuffer2, width * height * sizeof(glm::vec3));
  cudaMemset(dev_postBuffer2, 0, width * height * sizeof(glm::vec3));

  cudaFree(dev_depth);
  cudaMalloc(&dev_depth, width * height * sizeof(int));
  cudaFree(dev_mutex);
  cudaMalloc(&dev_mutex, width * height * sizeof(int));

  checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int *depth)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < w && y < h)
  {
    int index = x + (y * w);
    depth[index] = INT_MAX;
  }
}


/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int width = w / SSAA;
    int height = h / SSAA;
    int index = x + (y * width);

    if (x < width && y < height) {
        glm::vec3 color;
        for (int i = 0; i < SSAA; i++) {
          for (int j = 0; j < SSAA; j++) {
            int ind = (x *SSAA) + i + (y * SSAA + j) * w;
            color.x += glm::clamp(image[ind].x, 0.0f, 1.0f) * 255.0;
            color.y += glm::clamp(image[ind].y, 0.0f, 1.0f) * 255.0;
            color.z += glm::clamp(image[ind].z, 0.0f, 1.0f) * 255.0;
          }
        }
        color /= (SSAA*SSAA);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


/**
* Returns texture color for a fragment
*/
__device__
glm::vec3 getColor(Fragment &fragment, glm::vec2 uv) {
  // the color is stored in a 1D array of floats..
  //    convert to 1D index
  //    scale by 3

  int index = ((int)uv.x + (int)uv.y * fragment.texWidth) * 3;
  glm::vec3 col(fragment.dev_diffuseTex[index],
    fragment.dev_diffuseTex[index + 1],
    fragment.dev_diffuseTex[index + 2]);
  return col / 255.f; // map colors to 0-1 range..
}


/**
* Returns Bilinear Filtered texture color for a fragment
* Reference: https://en.wikipedia.org/wiki/Bilinear_filtering
*/
__device__
glm::vec3 getBilinearFilteredColor(Fragment &fragment, glm::vec2 uv) {
  // get 4 valid indices..
  int intX0 = uv.x;
  int intY0 = uv.y;
  int intX1 = glm::clamp(intX0 + 1, 0, fragment.texWidth - 1);
  int intY1 = glm::clamp(intY0 + 1, 0, fragment.texHeight - 1);

  // get colors at 4 texels..
  glm::vec3 col00 = getColor(fragment, glm::vec2(intX0, intY0));
  glm::vec3 col01 = getColor(fragment, glm::vec2(intX0, intY1));
  glm::vec3 col10 = getColor(fragment, glm::vec2(intX1, intY0));
  glm::vec3 col11 = getColor(fragment, glm::vec2(intX1, intY1));

  // lerp based on fractional parts..
  float fracX = uv.x - intX0;
  float fracY = uv.y - intY0;

  glm::vec3 col0001 = glm::mix(col00, col01, fracY);
  glm::vec3 col1011 = glm::mix(col10, col11, fracY);
  return glm::mix(col0001, col1011, fracX);
}


/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    if (x < w && y < h) {
      int index = x + (y * w);

      glm::vec3 &outPix = framebuffer[index];
      Fragment &frag = fragmentBuffer[index];

      if (frag.color == glm::vec3()) {
        outPix *= 0.f;
        return;
      }
		  
#if POINTS || WIREFRAME
      outPix = frag.color;

#elif DEBUG_DEPTH
      outPix = frag.color; // maybe scale this to look better???

#elif DEBUG_NORMALS
      outPix = frag.color;

#elif TEXTURE
      if (frag.dev_diffuseTex != NULL) {

  #if TEXTURE_BILINEAR_FILT
        glm::vec2 uv(frag.texcoord0.x * frag.texWidth, frag.texcoord0.y * frag.texHeight);
        outPix = getBilinearFilteredColor(frag, uv);
  #else
        glm::vec2 uv(frag.texcoord0.x * frag.texWidth, frag.texcoord0.y * frag.texHeight);
        outPix = getColor(frag, uv);
  #endif

        // LAMBERT SHADING:
        glm::vec3 lightDir = glm::normalize(frag.eyePos - glm::vec3(1.f));
        outPix *= glm::max(fabs(glm::dot(lightDir, frag.eyeNor)), 0.2f);
      }
      else {
        outPix = glm::vec3(); // reset color..
      }

#else

      // LAMBERT SHADING:

      glm::vec3 lightDir = frag.eyePos - glm::vec3(light[0], light[1], light[2]);
      lightDir = glm::normalize(lightDir);
      float dot = glm::dot(-lightDir, frag.eyeNor);
      if (dot <= 0.1f) {
        dot = 0.1f;
      }
      glm::vec3 col = frag.color * dot;
      outPix = col;

#endif
    }
}



// Screen space ambient occlusion
// https://www.gamedev.net/articles/programming/graphics/a-simple-and-practical-approach-to-ssao-r2753/
__global__
void shaderSSAO(int w, int h, Fragment *fragmentBuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < w && y < h) {
    int index = x + (y * w);

    Fragment &frag = fragmentBuffer[index];
    float ao = 0.f;
    for (int i = -2; i < 3; i++) {
      for (int j = -2; j < 3; j++) {
        int xi = x + i;
        int yi = y + i;

        if (xi > 0 && yi > 0 && xi < w && yi < h && i != 0 && j != 0) {
          int idxi = xi + (yi * w);
          Fragment &fragi = fragmentBuffer[idxi];
          glm::vec3 N = fragi.eyeNor;
          glm::vec3 V = fragi.eyePos - frag.eyePos;
          float d = glm::length(V);
          ao += glm::max(0.f, glm::dot(N, glm::normalize(V))) * (1.f / (1.f + d));
        }
      }
    }

    frag.color = ao * glm::vec3(1.f);
  }
}



/**
* Post Processing Shader
* operates on framebuffer
* uses the __constant__ gaussian kernel defined above..
*/
__global__
void postProcess(bool dirX, int w, int h, glm::vec3 *frameBuffer, int* depthBuffer, glm::vec3 *postBuffer) {
  int x = dirX ? threadIdx.x : blockIdx.x;
  int y = dirX ? blockIdx.x : threadIdx.x;
  int index = x + (y * w);

  if (x < w && y < h) {

    glm::vec3 col = mat[0] * frameBuffer[index];
    if (dirX) {
      for (int i = 1; i < 5; i++) {
        if (x + i < w) {
          col += mat[i] * frameBuffer[(x + i) + y * w];
        }
        if (x - i >= 0) {
          col += mat[i] * frameBuffer[(x - i) + y * w];
        }
      }
    }
    else {
      for (int i = 1; i < 5; i++) {
        if (y + i < h) {
          col += mat[i] * frameBuffer[x + (y + i) * w];
        }
        if (y - i >= 0) {
          col += mat[i] * frameBuffer[x + (y - i) * w];
        }
      }
    }

    postBuffer[index] = col;
  }
  else {
    postBuffer[index] = frameBuffer[index];
  }
}

__global__
void postProcessShared(bool dirX, int w, int h, glm::vec3 *frameBuffer, int* depthBuffer, glm::vec3 *postBuffer) {
  int x = dirX ? threadIdx.x : blockIdx.x;
  int y = dirX ? blockIdx.x : threadIdx.x;
  int index = x + (y * w);

  if (x >= w || y >= h) {
    return;
  }

  // https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
  extern __shared__ glm::vec3 sm[];
  if (dirX) {
    sm[x] = frameBuffer[index];
  }
  else {
    sm[y] = frameBuffer[index];
  }
  __syncthreads();

  int depth = depthBuffer[index];
  glm::vec3 col;
  if (dirX) {
    col = mat[0] * sm[x];
    for (int i = 1; i < 5; i++) {
      if (x + i < w) {
        if (abs(depthBuffer[(x + i) + y * w] - depth) < 10000) {
          col += mat[i] * sm[x + i];

        }
      }
      if (x - i >= 0) {
        if (abs(depthBuffer[(x - i) + y * w] - depth) < 10000) {
          col += mat[i] * sm[x - i];
        }
      }
    }
  }
  else {
    col = mat[0] * sm[y];
    for (int i = 1; i < 5; i++) {
      if (y + i < h) {
        if (abs(depthBuffer[x + (y + i) * w] - depth) < 10000) {
          col += mat[i] * sm[y + i];
        }
      }
      if (y - i >= 0) {
        if (abs(depthBuffer[x + (y - i) * w] - depth) < 10000) {
          col += mat[i] * sm[y - i];
        }
      }
    }
  }

  postBuffer[index] = col;
}


/**
* Blend kernel for DOF
* reference: https://mynameismjp.wordpress.com/the-museum/samples-tutorials-tools/depth-of-field-sample/
*/
__global__
void postBlend(int w, int h, glm::vec3 *frameBuffer, int *depthBuffer, glm::vec3 *postBuffer) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * w);

  if (x < w && y < h && frameBuffer[index]!=glm::vec3()) {
    float z = depthBuffer[index] * .0001f;
    //z = z < 0.3 ? fabs(z - 0.3) * 2.f : fabs(z - 0.3) * 2.f;
    //postBuffer[index] *= z;
    //postBuffer[index] += (1 - z) * frameBuffer[index];

    z = fabs(z - 0.5) * 2.f;
    postBuffer[index] *= z;
    postBuffer[index] += (1 - z) * frameBuffer[index];
  }
}



/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
#if BACK_FACE_CULLING
    cudaMalloc(&dev_primitives_compact, totalNumPrimitives * sizeof(Primitive));
#endif
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

    VertexOut &outVert = primitive.dev_verticesOut[vid];
    
    glm::vec4 inPos(primitive.dev_position[vid], 1.f);
    glm::vec4 outPos;

    outPos = MVP * inPos; // transform
    outPos /= outPos.w; // rehomogenize

    outVert.pos.x = (1.f - outPos.x) * width * 0.5f;
    outVert.pos.y = (1.f - outPos.y) * height * 0.5f;
    outVert.pos.z = -outPos.z;

    outVert.eyePos = glm::vec3(MV * inPos);
    outVert.eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);

#if TEXTURE

    outVert.texcoord0 = primitive.dev_texcoord0[vid];
    outVert.dev_diffuseTex = primitive.dev_diffuseTex;
    outVert.texWidth = primitive.diffuseTexWidth;
    outVert.texHeight = primitive.diffuseTexHeight;

#endif
		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
	}
}



static int curPrimitiveBeginId = 0;
__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
      VertexOut &vout = primitive.dev_verticesOut[primitive.dev_indices[iid]];
      dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
        = vout;

#if BACK_FACE_CULLING

      if (glm::dot(vout.eyeNor, glm::vec3(0.f, 0.f, 1.f)) < 0.f) {
        dev_primitives[pid + curPrimitiveBeginId].back = true;
      }
      else {
        dev_primitives[pid + curPrimitiveBeginId].back = false;
      }

#endif
		}

		// TODO: other primitive types (point, line)
    // point, line handled in rasterization..
	}
	
}




/**
* Creates a line from two points (A and B) using Digital Differential Analyzer (DDA) Algorithm
*
* Reference: http://www.geeksforgeeks.org/dda-line-generation-algorithm-computer-graphics/
*/
__device__
void drawLine(glm::vec3 A, glm::vec3 B, Fragment *fragBuf, int width) {
  float dx = B.x - A.x;
  float dy = B.y - A.y;
  float steps = fabs(dx) > fabs(dy) ? dx : dy;
  dx /= fabs(steps);
  dy /= fabs(steps);

  float x = A.x, y = A.y;
  for (int i = 0; i < steps; i++) {
    int index = (int)x + (int)y * width;
    fragBuf[index].color = glm::vec3(0.98);
    x += dx;
    y += dy;
  }
}



__global__
void _rasterizeTriangles(int numTris, Primitive *dev_primitives, Fragment *fragBuf, int *depthBuf, int *mutex, int width, int height) {

  // index id
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < numTris) {
    Primitive &prim = dev_primitives[tid];
    glm::vec3 tri[3] = { glm::vec3(prim.v[0].pos), 
                         glm::vec3(prim.v[1].pos), 
                         glm::vec3(prim.v[2].pos) };
    
#if BACK_FACE_CULLING_WITHOUT_COMPACTION
    if (glm::dot(prim.v[0].eyeNor, glm::vec3(0.f, 0.f, 1.f)) < 0.f) {
      return;
    }
#endif

#if POINTS

    for (int i = 0; i < 3; i++) {
      int x = tri[i].x;
      int y = tri[i].y;
      int index = x + (y * width);

  #if DEBUG_DEPTH
      fragBuf[index].color = glm::abs(glm::vec3(1.f + tri[i].z));
  #elif DEBUG_NORMALS
      fragBuf[index].color = prim.v[i].eyeNor;
  #else
      fragBuf[index].color = glm::vec3(0.98);
  #endif
    
    }

#elif WIREFRAME

    drawLine(tri[0], tri[1], fragBuf, width);
    drawLine(tri[1], tri[2], fragBuf, width);
    drawLine(tri[2], tri[0], fragBuf, width);

#else

    AABB bbox = getAABBForTriangle(tri);

#if BBOX_OPTIMIZATIONS
    // return when the entire bbox is outside screen..
    if (bbox.min.x > width || bbox.max.x<0 || bbox.min.y>height || bbox.max.y < 0) {
      return;
    }

    // clip bounding boxes to the screen size.. 
    // won't cause much divergence as most of the threads 
    // in a warp would fall in the same category most of the time..
    if (bbox.min.x < 0) {
      bbox.min.x = 0.f;
    }
    if (bbox.min.y < 0) {
      bbox.min.y = 0.f;
    }
    if (bbox.max.x > width) {
      bbox.min.x = width;
    }
    if (bbox.max.y > height) {
      bbox.min.x = height;
    }
    
#endif

    for (int y = bbox.min.y; y <= bbox.max.y; y++) {
      for (int x = bbox.min.x; x <= bbox.max.x; x++) {

        glm::vec2 point(x, y);
        glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, point);

        if (!isBarycentricCoordInBounds(baryCoord)) {
          continue;
        }

        int baryZ = getZAtCoordinate(baryCoord, tri) * 10000;

        int index = x + (y * width);

        glm::vec3 eyePosition = prim.v[0].eyePos * baryCoord.x
          + prim.v[1].eyePos * baryCoord.y
          + prim.v[2].eyePos * baryCoord.z;

        bool isSet;
        do {
          isSet = (atomicCAS(&mutex[index], 0, 1) == 0);
          if (isSet) {

            if (depthBuf[index] > baryZ) {

              depthBuf[index] = baryZ;

              fragBuf[index].eyePos = eyePosition;

              fragBuf[index].eyeNor = prim.v[0].eyeNor * baryCoord.x
                + prim.v[1].eyeNor * baryCoord.y
                + prim.v[2].eyeNor * baryCoord.z;

#if DEBUG_DEPTH

              fragBuf[index].color = glm::abs(glm::vec3(1.f - baryZ / 10000.f)); 

#elif DEBUG_NORMALS

              fragBuf[index].color = fragBuf[index].eyeNor;

#elif TEXTURE
#if TEXTURE_PERSP_CORRECT
              float z0 = prim.v[0].eyePos.z, z1 = prim.v[1].eyePos.z, z2 = prim.v[2].eyePos.z;
              float z = baryCoord.x / z0 + baryCoord.y / z1 + baryCoord.z / z2;
              fragBuf[index].texcoord0 = (prim.v[0].texcoord0 / z0 * baryCoord.x
                + prim.v[1].texcoord0 / z1 * baryCoord.y
                + prim.v[2].texcoord0 / z2 * baryCoord.z) / z;
#else
              fragBuf[index].texcoord0 = prim.v[0].texcoord0 * baryCoord.x
                + prim.v[1].texcoord0 * baryCoord.y
                + prim.v[2].texcoord0 * baryCoord.z;
#endif
              fragBuf[index].dev_diffuseTex = prim.v[0].dev_diffuseTex;
              fragBuf[index].texWidth = prim.v[0].texWidth;
              fragBuf[index].texHeight = prim.v[0].texHeight;

              fragBuf[index].color = glm::vec3(0.98);
#else // lambert
              fragBuf[index].color = glm::vec3(0.98);
#endif
            }
            mutex[index] = 0;
          }
        } while (!isSet);
      }
    }

#endif

  }
}


/**
* predicate struct for thrust::copy_if
*/
struct isNotBack {
  __host__ __device__
  bool operator()(const Primitive &prim) {
    return !prim.back;
  }
};

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly <<<numBlocksForVertices, numThreadsPerBlock>>> (p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly <<<numBlocksForIndices, numThreadsPerBlock>>>
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}

  cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
  cudaMemset(dev_mutex, 0, width * height * sizeof(int));
  initDepth << <blockCount2d, blockSize2d >> > (width, height, dev_depth);

#if BACK_FACE_CULLING

  cudaMemset(dev_primitives_compact, 0, totalNumPrimitives * sizeof(Primitive));
  Primitive* end = thrust::copy_if(thrust::device, dev_primitives, dev_primitives + totalNumPrimitives, dev_primitives_compact, isNotBack());
  checkCUDAError("Back face culling: thrust::partition");
  totalNumPrimitivesCompact = end - dev_primitives_compact;

  //// TODO: rasterize
  dim3 numThreadsPerBlock(128);
  dim3 numBlocksPerTriangle((totalNumPrimitivesCompact + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
  _rasterizeTriangles << <numBlocksPerTriangle, numThreadsPerBlock >> >
    (totalNumPrimitivesCompact, dev_primitives_compact, dev_fragmentBuffer, dev_depth, dev_mutex, width, height);
  checkCUDAError("rasterize triangles");

#else

  //// TODO: rasterize
  dim3 numThreadsPerBlock(128);
  dim3 numBlocksPerTriangle((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
  _rasterizeTriangles <<<numBlocksPerTriangle, numThreadsPerBlock>>>
    (totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex, width, height);
  checkCUDAError("rasterize triangles");


#endif

#if SSAO

  // SSAO
  shaderSSAO << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer);
  checkCUDAError("SSAO");

#endif

  // Copy fragmentbuffer colors into framebuffer [frame buffer is sent for post processing]
  render <<<blockCount2d, blockSize2d>>> (width, height, dev_fragmentBuffer, dev_framebuffer);
  checkCUDAError("fragment shader");

#if BLUR
  #if BLUR_SHARED

    // Doing seperate blur in X and then in Y to make things faster.. 
    // Using 2 postbuffers to avoid over-writing and collision between threads while doing blurring in Y dir..

    // blur in x dir
    dim3 threadsX(width), blocksX(height), threadsY(height), blocksY(width);
    bool dirX = true;
    postProcessShared <<<blocksX, threadsX, width * sizeof(glm::vec3)>>> (dirX, width, height, dev_framebuffer, dev_depth, dev_postBuffer1);
    checkCUDAError("post shader X");

    // blur in y dir
    dirX = false;
    postProcessShared <<<blocksY, threadsY, height * sizeof(glm::vec3)>>> (dirX, width, height, dev_postBuffer1, dev_depth, dev_postBuffer2);
    checkCUDAError("post shader Y");

    // 2nd pass for more blurring..
    dirX = true;
    postProcessShared <<<blocksX, threadsX, width * sizeof(glm::vec3)>>> (dirX, width, height, dev_postBuffer2, dev_depth, dev_postBuffer1);
    checkCUDAError("post shader X");

    // blur in y dir
    dirX = false;
    postProcessShared <<<blocksY, threadsY, height * sizeof(glm::vec3)>>> (dirX, width, height, dev_postBuffer1, dev_depth, dev_postBuffer2);
    checkCUDAError("post shader Y");

  #else

    // This version doesn't use shared memory. It is implemented only for the sake of comparison
    // blur in x dir
    dim3 threadsX(width), threadsY(height), blocksX(height), blocksY(width);
    bool dirX = true;
    postProcess <<<blocksX, threadsX>>> (dirX, width, height, dev_framebuffer, dev_depth, dev_postBuffer1);
    checkCUDAError("post shader X");

    // blur in y dir
    dirX = false;
    postProcess <<<blocksY, threadsY>>> (dirX, width, height, dev_postBuffer1, dev_depth, dev_postBuffer2);
    checkCUDAError("post shader Y");

  #endif

    // blend for DOF
    postBlend <<<blockCount2d, blockSize2d>>> (width, height, dev_framebuffer, dev_depth, dev_postBuffer2);
    checkCUDAError("copy post render result to pbo");

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO <<<blockCount2d, blockSize2d>>> (pbo, width, height, dev_postBuffer2);
    checkCUDAError("copy post render result to pbo");

#else

  blockSize2d = dim3(sideLength2d, sideLength2d);
  blockCount2d = dim3((width/SSAA - 1) / blockSize2d.x + 1, (height/SSAA - 1) / blockSize2d.y + 1);

  // Copy framebuffer into OpenGL buffer for OpenGL previewing
  sendImageToPBO <<<blockCount2d, blockSize2d>>> (pbo, width, height, dev_framebuffer);
  checkCUDAError("copy render result to pbo");

#endif
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

  auto it(mesh2PrimitivesMap.begin());
  auto itEnd(mesh2PrimitivesMap.end());
  for (; it != itEnd; ++it) {
    for (auto p = it->second.begin(); p != it->second.end(); ++p) {
      cudaFree(p->dev_indices);
      cudaFree(p->dev_position);
      cudaFree(p->dev_normal);
      cudaFree(p->dev_texcoord0);
      cudaFree(p->dev_diffuseTex);

      cudaFree(p->dev_verticesOut);


      //TODO: release other attributes and materials
    }
  }

  ////////////

  cudaFree(dev_primitives);
  dev_primitives = NULL;

  cudaFree(dev_fragmentBuffer);
  dev_fragmentBuffer = NULL;

  cudaFree(dev_framebuffer);
  dev_framebuffer = NULL;

  cudaFree(dev_postBuffer1);
  dev_postBuffer1 = NULL;

  cudaFree(dev_postBuffer2);
  dev_postBuffer2 = NULL;

  cudaFree(dev_depth);
  dev_depth = NULL;

  cudaFree(dev_mutex);
  dev_mutex = NULL;

  cudaFree(dev_primitives_compact);
  dev_primitives = NULL;

  checkCUDAError("rasterize Free");
}
