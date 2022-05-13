//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Scholtz Bálint András
// Neptun : A8O5M2
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"

//A program alapját a moodleben található karakteranimációs program adja.
//Külsõ forrást is használtam a Parabola GenVertexData függvényéhez, ezeket alább jelölöm:

//Parabolához használt források:
// - https://slideplayer.com/slide/9622862/31
// - https://mathinsight.org/parametrized_surface_normal_vector
// - Elõadásból az Implicit kvadratikus felületek rész kiegészítésnek a felsõ két linkhez

inline mat4 TransRotMatrix(float angle, vec3 w, vec3 t) {
	float c = cosf(angle), s = sinf(angle);
	w = normalize(w);
	return mat4(vec4(c * (1 - w.x * w.x) + w.x * w.x, w.x * w.y * (1 - c) + w.z * s, w.x * w.z * (1 - c) - w.y * s, 0),
				vec4(w.x * w.y * (1 - c) - w.z * s, c * (1 - w.y * w.y) + w.y * w.y, w.y * w.z * (1 - c) + w.x * s, 0),
				vec4(w.x * w.z * (1 - c) + w.y * s, w.y * w.z * (1 - c) - w.x * s, c * (1 - w.z * w.z) + w.z * w.z, 0),
				vec4(t.x, t.y, t.z, 1));
}

inline mat4 ScaleTransRotMatrix(float angle, vec3 w, vec3 t, vec3 sc) {
	float c = cosf(angle), s = sinf(angle);
	w = normalize(w);
	return mat4(vec4((c * (1 - w.x * w.x) + w.x * w.x) * sc.x, w.x * w.y * (1 - c) + w.z * s, w.x * w.z * (1 - c) - w.y * s, 0),
				vec4(w.x * w.y * (1 - c) - w.z * s, (c * (1 - w.y * w.y) + w.y * w.y) * sc.y, w.y * w.z * (1 - c) + w.x * s, 0),
				vec4(w.x * w.z * (1 - c) + w.y * s, w.y * w.z * (1 - c) - w.x * s, (c * (1 - w.z * w.z) + w.z * w.z) * sc.z, 0),
				vec4(t.x, t.y, t.z, 1));
}



//---------------------------
class PhongShader : public GPUProgram {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform vec3  wLiDir;       // light source direction 
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight;		    // light dir in world space

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight  = wLiDir;
		   wView   = wEye - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform vec3 kd, ks, ka; // diffuse, specular, ambient ref
		uniform vec3 La, Le;     // ambient and point sources
		uniform float shine;     // shininess for specular ref

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight;        // interpolated world sp illum dir
		in vec2 texcoord;
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			vec3 color = ka * La + (kd * cost + ks * pow(cosd,shine)) * Le;
			fragmentColor = vec4(color, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
};

PhongShader* gpuProgram; // vertex and fragment shaders

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = 1;
		fov = 80.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 100;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-wEye) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
	void SetUniform() {
		int location = glGetUniformLocation(gpuProgram->getId(), "wEye");
		if (location >= 0) glUniform3fv(location, 1, &wEye.x);
		else printf("uniform wEye cannot be set\n");
	}
};

Camera camera; // 3D camera

//-------------------------- -
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform() {
		gpuProgram->setUniform(kd, "kd");
		gpuProgram->setUniform(ks, "ks");
		gpuProgram->setUniform(ka, "ka");
		int location = glGetUniformLocation(gpuProgram->getId(), "shine");
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec3 wLightDir;

	Light() : La(1, 1, 1), Le(3, 3, 3) { }
	void SetUniform(bool enable) {
		if (enable) {
			gpuProgram->setUniform(La, "La");
			gpuProgram->setUniform(Le, "Le");
		}
		else {
			gpuProgram->setUniform(vec3(0,0,0), "La");
			gpuProgram->setUniform(vec3(0,0,0), "Le");
		}
		gpuProgram->setUniform(wLightDir, "wLiDir");
	}
};



//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
	unsigned int vao, type;        // vertex array object
protected:
	int nVertices;
public:
	Geometry(unsigned int _type) {
		type = _type;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw(mat4 M, mat4 Minv) {
		mat4 MVP = M * camera.V() * camera.P();
		gpuProgram->setUniform(MVP, "MVP");
		gpuProgram->setUniform(M, "M");
		gpuProgram->setUniform(Minv, "Minv");
		glBindVertexArray(vao);
		glDrawArrays(type, 0, nVertices);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
public:
	ParamSurface() : Geometry(GL_TRIANGLES) {}

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 16, int M = 16) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVertices = N * M * 6;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				vtxData.push_back(GenVertexData((float)i / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)j / M));
				vtxData.push_back(GenVertexData((float)(i + 1) / N, (float)(j + 1) / M));
				vtxData.push_back(GenVertexData((float)i / N, (float)(j + 1) / M));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVertices * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
};

//---------------------------
class SphereRaw : public ParamSurface {
	//---------------------------
	float r;
public:
	SphereRaw(float _r) {
		r = _r;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cosf(u * 2.0f * M_PI) * sin(v * M_PI), sinf(u * 2.0f * M_PI) * sinf(v * M_PI), cosf(v * M_PI));
		vd.position = vd.normal * r;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Sphere {
	//---------------------------
	Material* material;
	SphereRaw* sphere;
public:
	Sphere(float _r, Material* _m) {
		material = _m;
		sphere = new SphereRaw(_r);
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		sphere->Draw(M, Minv);
	}
};

//---------------------------
class CylinderRaw : public ParamSurface {
	//---------------------------
	float r, height;
public:
	CylinderRaw(float _r, float _height) {
		r = _r;
		height = _height;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * 2 * M_PI, V = v * height;
		vec3 base(cos(U) * r, sin(U) * r, 0), spine(0, 0, V);
		vd.position = base + spine;
		vd.texcoord = vec2(u,v);
		vd.normal = base;
		return vd;
	}
};

//---------------------------
class Cylinder {
	//---------------------------
	Material* material;
	CylinderRaw* cylinder;
public:
	Cylinder(float _r, float _height, Material* _m) {
		material = _m;
		cylinder = new CylinderRaw(_r, _height);
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		cylinder->Draw(M, Minv);
	}
};



//---------------------------
class ParaboloidRaw : public ParamSurface {
	//---------------------------
	float height, width;
public:
	ParaboloidRaw(float _height, float _width) {
		height = _height;
		width = _width;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u * height;
		float V = v * 2.0f * M_PI;
		float x = U * cosf(V) * width;
		float y = U * sinf(V) * width;
		float z = U * U;
		vec3 du = vec3(cosf(V), sinf(V), U);
		vec3 dv = vec3(-U * sinf(V), U * cosf(V), 0);
		vd.normal = normalize(cross(du, dv));
		vd.position = vec3(x, y, z);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Paraboloid {
	//---------------------------
	Material* material;
	ParaboloidRaw* para;
public:
	Paraboloid(float _r, float _height, Material* _m) {
		material = _m;
		para = new ParaboloidRaw(_r, _height);
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		para->Draw(M, Minv);
	}
};

//---------------------------
class CircleRaw : public ParamSurface {
	//---------------------------
	float r, height;
public:
	CircleRaw(float _r) {
		r = _r;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0, 1, 0);
		vd.position = vec3(u * cos(v * 2.0f * M_PI), u * sin(v * 2.0f * M_PI)) * r;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Circle {
	//---------------------------
	Material* material;
	CircleRaw* circle;
public:
	Circle(float _r, Material* _m) {
		material = _m;
		circle = new CircleRaw(_r);
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		circle->Draw(M, Minv);
	}
};


//---------------------------
class Quad : public ParamSurface {
	//---------------------------
	float size;
public:
	Quad() {
		size = 100;
		Create(20, 20);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0, 1, 0);
		vd.position = vec3((u - 0.5) * 2, 0, (v - 0.5) * 2) * size;
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
class Floor {
	//---------------------------
	Material* material;
	Geometry* quad;
public:
	Floor(Material* _m) {
		material = _m;
		quad = new Quad();
	}
	void Draw(mat4 M, mat4 Minv) {
		material->SetUniform();
		quad->Draw(M, Minv);
	}
};

const float boneRadius = 0.5;
const float legLength = 5;

#define INVERSE_KINEMATICS
//===============================================================


//---------------------------
class Scene {
	//---------------------------
	Floor* floor;
	Sphere* sph1;
	Cylinder* cyl1;
	Sphere* sph2;
	Cylinder* cyl2;
	Sphere* sph3;
	Cylinder* cyl3;
	Paraboloid* para;
	Circle* circle;
	float frame = 0;
public:
	Light light;

	void Build() {
		// Materials
		Material* material0 = new Material;
		material0->kd = vec3(0.2f, 0.3f, 1);
		material0->ks = vec3(1, 1, 1);
		material0->ka = vec3(0.2f, 0.3f, 1);
		material0->shininess = 20;

		Material* material1 = new Material;
		material1->kd = vec3(0, 1, 1);
		material1->ks = vec3(2, 2, 2);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 200;

		// Geometries
		floor = new Floor(material1);
		sph1 = new Sphere(1, material0);
		cyl1 = new Cylinder(5, 4, material0);
		sph2 = new Sphere(1, material0);
		cyl2 = new Cylinder(0.5, 6, material0);
		sph3 = new Sphere(1, material0);
		cyl3 = new Cylinder(0.5, 6, material0);
		para = new Paraboloid(2, 2, material0);
		circle = new Circle(5, material0);


		// Camera
		camera.wEye = vec3(0, 0, 4);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Light
		light.wLightDir = vec3(5, 5, 4);

	}
	void Render() {
		camera.SetUniform();
		light.SetUniform(true);

		mat4 unit = TranslateMatrix(vec3(0, 0, 0));
		floor->Draw(unit, unit);

		//Segment Rotation/Position
		mat4 segment1RotateMtx = RotationMatrix(frame * M_PI/3, vec3(1, 4, 0));
		mat4 Invsegment1RotateMtx = RotationMatrix(-frame * M_PI /3, -vec3(1, 4, 0));
		mat4 segment2RotateMtx = RotationMatrix(frame * M_PI/3, vec3(1, 1, 0));
		mat4 segment3RotateMtx = RotationMatrix(frame * M_PI/3, vec3(1, 4, 4));
		mat4 segment1Position = TranslateMatrix(vec3(0, 4, 0));
		mat4 segment2Position = TranslateMatrix(vec3(0, 6, 0));
		mat4 segment3Position = TranslateMatrix(vec3(0, 6, 0));

		//Talp gyuru
		mat4 cyl1TransRotMatrix = TransRotMatrix(M_PI / 2, vec3(-1, 0, 0), vec3(0, 0, 0));
		mat4 Invcyl1TransRotMatrix = cyl1TransRotMatrix * TransRotMatrix(-M_PI / 2, -vec3(-1, 0, 0), vec3(0, 0, 0));
		cyl1->Draw(cyl1TransRotMatrix, Invcyl1TransRotMatrix);

		//Talp teto
		mat4 crcTransRotMatrix = TransRotMatrix(M_PI / 2, vec3(-1, 0, 0), vec3(0, 4, 0));
		mat4 InvcrcTransRotMatrix = crcTransRotMatrix * TransRotMatrix(M_PI / 2, -vec3(-1, 0, 0), -vec3(0, 4, 0));
		circle->Draw(crcTransRotMatrix, InvcrcTransRotMatrix);

		//1. gomb
		mat4 sph1unit = TransRotMatrix(0, vec3(0, 1, 0), vec3(0, 4, 0));
		sph1->Draw(sph1unit, sph1unit);


		//1 rud
		mat4 cyl2RotMatrix = RotationMatrix(M_PI / 2, vec3(-1, 0, 0));
		mat4 cyl2TransRotMatrix = TranslateMatrix(vec3(0, 4, 0));
		cyl2TransRotMatrix = cyl2RotMatrix * segment1RotateMtx * cyl2TransRotMatrix;
		mat4 Invcyl2RotMatrix = cyl2TransRotMatrix * RotationMatrix(-M_PI / 2, -vec3(-1, 0, 0)) * Invsegment1RotateMtx;
		cyl2->Draw(cyl2TransRotMatrix, Invcyl2RotMatrix);

		//2. gomb
		mat4 sph2unit = TranslateMatrix(vec3(0, 6, 0));
		sph2unit = sph2unit * segment1RotateMtx * segment1Position;
		sph2->Draw(sph2unit, sph2unit);

		//2 rud
		mat4 cyl3RotMatrix = RotationMatrix(M_PI / 2, vec3(-1, 0, 0));
		mat4 cyl3TransRotMatrix = TranslateMatrix(vec3(0, 6, 0));
		cyl3TransRotMatrix = cyl3RotMatrix * segment2RotateMtx * cyl3TransRotMatrix * segment1RotateMtx * segment1Position;
		//cyl3TransRotMatrix = segment1RotateMtx * segment1Position * segment2RotateMtx * cyl3TransRotMatrix;
		cyl2->Draw(cyl3TransRotMatrix, cyl3TransRotMatrix);

		//3. gomb
		mat4 sph3unit = TranslateMatrix(vec3(0, 6, 0));
		sph3unit = sph3unit * segment2RotateMtx * segment2Position * segment1RotateMtx * segment1Position;
		sph3->Draw(sph3unit, sph3unit);

		//Paraboloid
		mat4 parunit = segment3RotateMtx * segment3Position * segment2RotateMtx * segment2Position * segment1RotateMtx * segment1Position;
		para->Draw(parunit, parunit);



		// shadow matrix that projects the man onto the floor

		light.SetUniform(false);
		
		mat4 shadowMatrix = { 1, 0, 0, 0,
							-light.wLightDir.x / light.wLightDir.y, 0, -light.wLightDir.z / light.wLightDir.y, 0,
							0, 0, 1, 0,
							0, 0.001f, 0, 1 };

		cyl1->Draw(cyl1TransRotMatrix * shadowMatrix, cyl1TransRotMatrix * shadowMatrix);
		cyl2->Draw(cyl2TransRotMatrix * shadowMatrix, cyl2TransRotMatrix * shadowMatrix);
		cyl3->Draw(cyl3TransRotMatrix * shadowMatrix, cyl3TransRotMatrix * shadowMatrix);
		sph1->Draw(sph1unit * shadowMatrix, sph1unit * shadowMatrix);
		sph2->Draw(sph2unit * shadowMatrix, sph2unit * shadowMatrix);
		sph3->Draw(sph3unit * shadowMatrix, sph3unit * shadowMatrix);
		para->Draw(parunit * shadowMatrix, parunit * shadowMatrix);
	}

	void Animate(float t) {
		static float tprev = 0;
		float dt = t - tprev;
		tprev = t;

		//pman->Animate(dt);

		frame++;

		static float cam_angle = 0;
		cam_angle += 0.01 * dt;			// camera rotate

		const float camera_rad = 30;
		camera.wEye = vec3(cos(cam_angle) * camera_rad, 20, sin(cam_angle) * camera_rad);
		camera.wLookat = vec3(0, 0,	0);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	gpuProgram = new PhongShader();
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 30.0f;				// convert msec to sec
	scene.Animate(sec);					// animate the camera
	glutPostRedisplay();					// redraw the scene
}