#include "minipbrt.h"
#include "minipbrt.h"
#include "Eigen"
#include <fstream>
#include <string>
#include <iostream>
#include <assimp/scene.h>
std::string getTextureString() {
	std::string out = "=let\n";
	out += "{\n";
	out += "base::texture_return tint_mono = base::file_texture(\n";
	out += "									 texture: parTexture,\n";
	out += "									 color_scale : color(1.0),\n";
	out += "									 color_offset : color(0.0),\n";
	out += "									 uvw : base::transform_coordinate(\n";
	out += "												transform : base::rotation_translation_scale(rotation : parRotation, translation : parTranslation, scaling : parScaling),\n";
	out += "												coordinate : base::coordinate_source(coordinate_system : base::texture_coordinate_uvw, texture_space : 0)\n";
	out += "											  ),\n";
	out += "									 mono_source : base::mono_average\n";
	out += "								   );\n";
	out += "} in material(\n";
	return out;
}
std::pair<std::string, std::string> producePlasticMDL(std::string outfileMtl, minipbrt::PlasticMaterial* toExport, int number) {
	Eigen::Vector3f kD = Eigen::Vector3f(toExport->Kd.value);
	Eigen::Vector3f kS = Eigen::Vector3f(toExport->Ks.value);
	float roughness = toExport->roughness.value;
	std::string expString = "mdl 1.7;\nimport ::df::*;\n";
	expString += "\nexport material " + outfileMtl + "_" + "plastic_" + std::to_string(number) + "(\n";
	int count = 2;
	float weights = 1.f / float(count);
	for (int i = 0; i < count; i++) {
		expString += "  uniform float parWeight" + std::to_string(i) + "          =" + std::to_string(weights) + ",\n";
	}
	std::string mode = "df::scatter_reflect";
	expString += "  uniform color parDiffuseTint = color(" + std::to_string(kD[0]) + ", " + std::to_string(kD[1]) + ", " + std::to_string(kD[2]) + ")";
	expString += ",\n  uniform float parDiffuseRoughness = " + std::to_string(roughness) + ",\n";
	expString += "  uniform color parSpecularTint = color(" + std::to_string(kS[0]) + ", " + std::to_string(kS[1]) + ", " + std::to_string(kS[2]) + ")\n";
	expString += ")\n=\n";
	expString += "material(\n";
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::clamped_mix(\n";
	expString += "      components: df::bsdf_component[](\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight0,\n";
	expString += "	        component: df::specular_bsdf(\n";
	expString += "	          tint : parSpecularTint,\n";
	expString += "	          mode: " + mode + "\n          )\n        ),\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight1,\n";
	expString += "	        component: df::diffuse_reflection_bsdf(\n";
	expString += "	          tint : parDiffuseTint";
	expString += ",\n	          roughness : parDiffuseRoughness\n";
	expString += "          )\n        )\n      )\n    )\n  )\n);";
	std::ofstream out(outfileMtl + "_plastic" + std::to_string(number) + ".mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_plastic" + std::to_string(number) + ".mdl", outfileMtl + "_" + "plastic_" + std::to_string(number));
}
std::pair<std::string, std::string> produceMatteMDL(std::string outfileMtl, minipbrt::MatteMaterial* toExport, int number, std::vector<std::string> textureLocs) {
	Eigen::Vector3f kD = Eigen::Vector3f(toExport->Kd.value);
	// Only contains diffuse + roughness component
	float roughness = toExport->sigma.value;
	float texture = toExport->Kd.texture;
	texture = (texture == 4294967295 || textureLocs[texture] == "none") ? 4294967295 : texture;
	std::string expString = "mdl 1.7;\nimport ::df::*;\n";
	if (texture != 4294967295) {
		expString += "import ::tex::*;\nimport ::base::*;\n";
	}
	expString += "\nexport material " + outfileMtl + "_" + "matte_" + std::to_string(number) + "(\n";
	expString += "  uniform color parDiffuseTint = color(" + std::to_string(kD[0]) + ", " + std::to_string(kD[1]) + ", " + std::to_string(kD[2]) + "),\n";
	if (texture != 4294967295) {
		expString += " uniform texture_2d parTexture = texture_2d(\"./" + textureLocs[texture] + "\", tex::gamma_srgb), \n";
		expString += "  uniform float3 parRotation    = float3(0.0),\n";
		expString += "  uniform float3 parTranslation = float3(0.0, 0.0, 0.0),\n";
		expString += "  uniform float3 parScaling     = float3(1.0),\n";
	}
	expString += " uniform float parDiffuseRoughness = " + std::to_string(roughness) + "\n)\n";
	if (texture == 4294967295) {
		expString += "=\nmaterial(\n";
	}
	else {
		expString += getTextureString();
	}
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::diffuse_reflection_bsdf(\n";
	if (texture == 4294967295) {
		expString += "	          tint : parDiffuseTint,\n";
	}
	else {
		expString += "	          tint : parDiffuseTint * tint_mono.tint,\n";
	}
	expString += "      roughness: parDiffuseRoughness\n";
	expString += "    )\n  )\n);\n";
	std::ofstream out(outfileMtl + "_matte" + std::to_string(number) + ".mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_matte" + std::to_string(number) + ".mdl", outfileMtl + "_" + "matte_" + std::to_string(number));

}
std::pair<std::string, std::string> produceMirrorMDL(std::string outfileMtl, minipbrt::MirrorMaterial* toExport, int number) {
	Eigen::Vector3f kS = Eigen::Vector3f(toExport->Kr.value);
	std::string expString = "mdl 1.7;\nimport ::df::*;\n";
	expString += "\nexport material " + outfileMtl + "_" + "mirror" + std::to_string(number) + "(\n";
	std::string mode = "df::scatter_reflect";
	expString += "  uniform color parSpecularTint = color(" + std::to_string(kS[0]) + ", " + std::to_string(kS[1]) + ", " + std::to_string(kS[2]) + ")\n";
	expString += ")\n=\n";
	expString += "material(\n";
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::specular_bsdf(\n";
	expString += "      tint:      parSpecularTint,\n";
	expString += "      mode: " + mode;
	expString += "    )\n  )\n);\n";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "mirror.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "mirror.mdl", outfileMtl + "_" + "mirror" + std::to_string(number));
}
std::pair<std::string, std::string> produceMetalMDL(std::string outfileMtl, minipbrt::MetalMaterial* toExport, int number) {
	Eigen::Vector3f IoR = Eigen::Vector3f(toExport->eta.value);
	Eigen::Vector3f kRef = Eigen::Vector3f(toExport->k.value);
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::anno::*;\n";
	float uRoughness = toExport->uroughness.value;
	float vRoughness = toExport->vroughness.value;
	expString += "\nexport material " + outfileMtl + "_" + "metal" + std::to_string(number) + "(\n";
	expString += "  uniform color parIOR = color(" + std::to_string(IoR[0]) + ", " + std::to_string(IoR[1]) + ", " + std::to_string(IoR[2]) + "),\n";
	expString += "  uniform float parGlossyRoughnessU = " + std::to_string(uRoughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessV = " + std::to_string(vRoughness) + ",\n";
	expString += "  uniform color parGlossyTint = color(.95)\n";
	expString += ")\n=\n";
	expString += "material(\n";
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::simple_glossy_bsdf(\n";
	expString += "      roughness_u:      parGlossyRoughnessU,\n";
	expString += "      roughness_v:      parGlossyRoughnessU,\n";
	expString += "      tint :      parGlossyTint,\n";
	expString += "      mode : df::scatter_reflect";
	expString += "    )\n  )\n);\n";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "metal.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "metal.mdl", outfileMtl + "_" + "metal" + std::to_string(number));
}
std::pair<std::string, std::string> produceGlassMDL(std::string outfileMtl, minipbrt::GlassMaterial* toExport, int number) {
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::state::*;\nimport ::math::*;\nimport ::base::*;\nimport ::tex::*;\nimport ::anno::*;\n";
	Eigen::Vector3f cS = Eigen::Vector3f(toExport->Kr.value);
	Eigen::Vector3f cT = Eigen::Vector3f(toExport->Kr.value);
	expString += "\nexport material " + outfileMtl + "_" + "glass" + std::to_string(number) + "(\n";
	expString += "color transmission_color = color(" + std::to_string(cT[0]) + ", " + std::to_string(cT[1]) + ", " + std::to_string(cT[2]) + "),\n";
	expString += "color glass_color = color(" + std::to_string(cS[0]) + ", " + std::to_string(cS[1]) + ", " + std::to_string(cS[2]) + "),\n";
	expString += "float roughness = " + std::to_string(sqrt(toExport->uroughness.value)) + ",\n";
	expString += "uniform float ior = " + std::to_string(toExport->eta.value) + ",\n";
	expString += "uniform float base_thickness = .1,\n";
	expString += "uniform float abbe_number = 0.0,\n";
	expString += "float3 normal = state::normal()\n)\n=";
	expString += " let{\n";
	expString += "  bsdf glossy_bsdf = df::microfacet_ggx_smith_bsdf(\n";
	expString += "    mode: df::scatter_reflect,\n";
	expString += "    tint: color(1.),\n";
	expString += "    roughness_u : roughness * roughness\n";
	expString += "  );\n";
	expString += "  bsdf glossy_bsdf_transmission = df::microfacet_ggx_smith_bsdf(\n";
	expString += "    mode: df::scatter_transmit,\n";
	expString += "    tint: transmission_color,\n";
	expString += "    roughness_u: roughness*roughness\n";
	expString += "  );\n";
	expString += " } in material(\n";
	expString += "     thin_walled: false,\n";
	expString += "     surface : material_surface(\n";
	expString += "         df::weighted_layer(\n";
	expString += "             normal: normal,\n";
	expString += "             weight: 1.,\n";
	expString += "             layer: df::fresnel_layer(\n";
	expString += "                 layer : glossy_bsdf,\n";
	expString += "                 base: glossy_bsdf_transmission,\n";
	expString += "                 ior: ior,\n";
	expString += "                 normal: normal\n";
	expString += "             )\n";
	expString += "         )\n";
	expString += "     ),\n";
	expString += "    volume: material_volume(\n";
	expString += "      absorption_coefficient: (base_thickness <= 0)? color(0): math::log(glass_color) / -base_thickness\n";
	expString += "    ),\n";
	expString += "   ior: abbe_number == 0.0?color(ior):base::abbe_number_ior(ior, abbe_number)\n";
	expString += " ); ";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "glass.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "glass.mdl", outfileMtl + "_" + "glass" + std::to_string(number));
}
std::pair<std::string, std::string> produceUberMDL(std::string outfileMtl, minipbrt::UberMaterial* toExport, int number, std::vector<std::string> textureLocs){
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::anno::*;\n";
	Eigen::Vector3f kD = Eigen::Vector3f(toExport->Kd.value);
	Eigen::Vector3f kS = Eigen::Vector3f(toExport->Ks.value);
	Eigen::Vector3f kR = Eigen::Vector3f(toExport->Kr.value);
	Eigen::Vector3f kT = Eigen::Vector3f(toExport->Kt.value);
	float roughness = toExport->remaproughness;
	float uroughness = toExport->uroughness.value;
	float vroughness = toExport->vroughness.value;
	float texture = toExport->Kd.texture;
	texture = (texture == 4294967295 || textureLocs[texture] == "none") ? 4294967295 : texture;
	if (texture != 4294967295) {
		expString += "import ::tex::*;\nimport ::base::*;\n";
	}
	expString += "\nexport material " + outfileMtl + "_" + "uber_" + std::to_string(number) + "(\n";
	int count = 3;
	float weights = 0.33f;
	for (int i = 0; i < count; i++) {
		expString += "  uniform float parWeight" + std::to_string(i) + "          =" + std::to_string(weights) + ",\n";
	}
	if (texture != 4294967295) {
		expString += " uniform texture_2d parTexture = texture_2d(\"./" + textureLocs[texture] + "\", tex::gamma_srgb), \n";
		expString += "  uniform float3 parRotation    = float3(0.0),\n";
		expString += "  uniform float3 parTranslation = float3(0.0, 0.0, 0.0),\n";
		expString += "  uniform float3 parScaling     = float3(1.0),\n";
	}
	expString += "  uniform color parDiffuseTint = color(" + std::to_string(kD[0]) + ", " + std::to_string(kD[1]) + ", " + std::to_string(kD[2]) + "),\n";
	expString += "  uniform float parDiffuseRoughness = " + std::to_string(roughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessU = " + std::to_string(uroughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessV = " + std::to_string(vroughness) + ",\n";
	expString += "  uniform color parSpecularTint = color(" + std::to_string(kR[0]) + ", " + std::to_string(kR[1]) + ", " + std::to_string(kR[2]) + "),\n";
	expString += "  uniform color parGlossyTint = color(" + std::to_string(kS[0]) + ", " + std::to_string(kS[1]) + ", " + std::to_string(kS[2]) + ")\n)\n";
	if (texture == 4294967295) {
		expString += "=\nmaterial(\n";
	}
	else {
		expString += getTextureString();
	}
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::clamped_mix(\n";
	expString += "      components: df::bsdf_component[](\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight0,\n";
	expString += "	        component: df::specular_bsdf(\n";
	expString += "	          tint : parSpecularTint,\n";
	expString += "	          mode: df::scatter_reflect\n          )\n        ),\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight1,\n";
	expString += "	        component: df::diffuse_reflection_bsdf(\n";
	if (texture == 4294967295) {
		expString += "	          tint : parDiffuseTint,\n";
	}
	else {
		expString += "	          tint : parDiffuseTint * tint_mono.tint,\n";
	}
	expString += "	          roughness : parDiffuseRoughness\n          )\n        ),\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight2,\n";
	expString += "	        component: df::simple_glossy_bsdf(\n";
	expString += "	          roughness_u: parGlossyRoughnessU,\n";
	expString += "	          roughness_v: parGlossyRoughnessV,\n";
	expString += "	          tint: parGlossyTint,\n";
	expString += "	          mode: df::scatter_reflect\n";
	expString += "          )\n        )\n      )\n    )\n  )\n);";
	std::ofstream out(outfileMtl + "_uber_" + std::to_string(number) + ".mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_uber_" + std::to_string(number) + ".mdl", outfileMtl + "_" + "uber_" + std::to_string(number));
}
std::pair<std::string, std::string> produceSubstrateMDL(std::string outfileMtl, minipbrt::SubstrateMaterial* toExport, int number, std::vector<std::string> textureLocs) {
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::anno::*;\n";
	Eigen::Vector3f kD = Eigen::Vector3f(toExport->Kd.value);
	Eigen::Vector3f kS = Eigen::Vector3f(toExport->Ks.value);
	float roughness = toExport->remaproughness;
	float uroughness = toExport->uroughness.value;
	float vroughness = toExport->vroughness.value;
	float texture = toExport->Kd.texture;
	texture = (texture == 4294967295 || textureLocs[texture] == "none") ? 4294967295 : texture;
	if (texture != 4294967295) {
		expString += "import ::tex::*;\nimport ::base::*;\n";
	}
	expString += "\nexport material " + outfileMtl + "_" + "substrate_" + std::to_string(number) + "(\n";
	int count = 2;
	float weights = 1.f / float(count);
	for (int i = 0; i < count; i++) {
		expString += "  uniform float parWeight" + std::to_string(i) + "          =" + std::to_string(weights) + ",\n";
	}
	if (texture != 4294967295) {
		expString += " uniform texture_2d parTexture = texture_2d(\"./" + textureLocs[texture] + "\", tex::gamma_srgb), \n";
		expString += "  uniform float3 parRotation    = float3(0.0),\n";
		expString += "  uniform float3 parTranslation = float3(0.0, 0.0, 0.0),\n";
		expString += "  uniform float3 parScaling     = float3(1.0),\n";
	}
	expString += "  uniform color parDiffuseTint = color(" + std::to_string(kD[0]) + ", " + std::to_string(kD[1]) + ", " + std::to_string(kD[2]) + "),\n";
	expString += "  uniform float parDiffuseRoughness = " + std::to_string(roughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessU = " + std::to_string(uroughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessV = " + std::to_string(vroughness) + ",\n";
	expString += "  uniform color parGlossyTint = color(" + std::to_string(kS[0]) + ", " + std::to_string(kS[1]) + ", " + std::to_string(kS[2]) + ")\n";
	expString += ")\n";
	if (texture == 4294967295) {
		expString += "=\nmaterial(\n";
	}
	else {
		expString += getTextureString();
	}
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::clamped_mix(\n";
	expString += "      components: df::bsdf_component[](\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight0,\n";
	expString += "	        component: df::diffuse_reflection_bsdf(\n";
	if (texture == 4294967295) {
		expString += "	          tint : parDiffuseTint,\n";
	}
	else {
		expString += "	          tint : parDiffuseTint * tint_mono.tint,\n";
	}
	expString += "	          roughness : parDiffuseRoughness\n          )\n        ),\n";
	expString += "        df::bsdf_component(\n";
	expString += "          weight: parWeight1,\n";
	expString += "	        component: df::simple_glossy_bsdf(\n";
	expString += "	          roughness_u: parGlossyRoughnessU,\n";
	expString += "	          roughness_v: parGlossyRoughnessV,\n";
	expString += "	          tint: parGlossyTint,\n";
	expString += "	          mode: df::scatter_reflect\n";
	expString += "          )\n        )\n      )\n    )\n  )\n);";
	std::ofstream out(outfileMtl + "_substrate_" + std::to_string(number) + ".mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_substrate_" + std::to_string(number) + ".mdl", outfileMtl + "_" + "substrate_" + std::to_string(number));
}
std::pair<std::string, std::string> produceFourierMDL(std::string outfileMtl, minipbrt::FourierMaterial* toExport, int number) {
	Eigen::Vector3f IoR = Eigen::Vector3f(.5, .5, .5);
	Eigen::Vector3f kRef = Eigen::Vector3f(.5, .5, .5);
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::anno::*;\n";
	float uRoughness = .01f;
	float vRoughness = .01f;
	expString += "\nexport material " + outfileMtl + "_" + "fourier" + std::to_string(number) + "(\n";
	expString += "  uniform color parIOR = color(" + std::to_string(IoR[0]) + ", " + std::to_string(IoR[1]) + ", " + std::to_string(IoR[2]) + "),\n";
	expString += "  uniform float parGlossyRoughnessU = " + std::to_string(uRoughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessV = " + std::to_string(vRoughness) + ",\n";
	expString += "  uniform color parGlossyTint = color(" + std::to_string(kRef[0]) + ", " + std::to_string(kRef[1]) + ", " + std::to_string(kRef[2]) + ")\n";
	expString += ")\n=\n";
	expString += "material(\n";
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::simple_glossy_bsdf(\n";
	expString += "      roughness_u:      parGlossyRoughnessU,\n";
	expString += "      roughness_v:      parGlossyRoughnessU,\n";
	expString += "      tint :      parGlossyTint\n";
	expString += "      mode : df::scatter_reflect_transmit";
	expString += "    )\n  )\n);\n";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "fourier.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "fourier.mdl", outfileMtl + "_" + "fourier" + std::to_string(number));
}
std::pair<std::string, std::string> produceMixed(std::string outfileMtl, minipbrt::MixMaterial* toExport, int number) {
	Eigen::Vector3f IoR = Eigen::Vector3f(.5, .5, .5);
	Eigen::Vector3f kRef = Eigen::Vector3f(.5, .5, .5);
	std::string expString = "mdl 1.7;\nimport ::df::*;\nimport ::anno::*;\n";
	float uRoughness = .01f;
	float vRoughness = .01f;
	expString += "\nexport material " + outfileMtl + "_" + "mixed" + std::to_string(number) + "(\n";
	expString += "  uniform color parIOR = color(" + std::to_string(IoR[0]) + ", " + std::to_string(IoR[1]) + ", " + std::to_string(IoR[2]) + "),\n";
	expString += "  uniform float parGlossyRoughnessU = " + std::to_string(uRoughness) + ",\n";
	expString += "  uniform float parGlossyRoughnessV = " + std::to_string(vRoughness) + ",\n";
	expString += "  uniform color parGlossyTint = color(" + std::to_string(kRef[0]) + ", " + std::to_string(kRef[1]) + ", " + std::to_string(kRef[2]) + ")\n";
	expString += ")\n=\n";
	expString += "material(\n";
	expString += "  surface: material_surface(\n";
	expString += "    scattering: df::simple_glossy_bsdf(\n";
	expString += "      roughness_u:      parGlossyRoughnessU,\n";
	expString += "      roughness_v:      parGlossyRoughnessU,\n";
	expString += "      tint :      parGlossyTint,\n";
	expString += "      mode : df::scatter_reflect_transmit";
	expString += "    )\n  )\n);\n";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "mixed.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "mixed.mdl", outfileMtl + "_" + "mixed" + std::to_string(number));
}
std::string produceObjFromTriMesh(minipbrt::TriangleMesh *inp, std::string prefix) {
	std::vector<Eigen::Vector3f> verts;
	std::vector<Eigen::Vector3i> faces;
	std::vector<Eigen::Vector3f> norms;
	for (int i = 0; i < int(inp->num_indices / 3); i++) {
		faces.push_back(Eigen::Vector3i(inp->indices[3 * i], inp->indices[3 * i + 1], inp->indices[3 * i + 2]));
	}
	for (int i = 0; i < inp->num_vertices; i++) {
		Eigen::Vector3f vert = Eigen::Vector3f(inp->P[3 * i], inp->P[3 * i + 1], inp->P[3 * i + 2]);
		verts.push_back(vert);
	}
	std::ofstream out(prefix);
	for (int i = 0; i < verts.size(); i++) {
		out << "v " + std::to_string(verts[i][0]) + " " + std::to_string(verts[i][1]) + " " + std::to_string(verts[i][2]) + "\n";
	}
	out << "\n";
	for (int i = 0; i < faces.size(); i++) {
		out << "f " + std::to_string(faces[i][0] + 1) + " " + std::to_string(faces[i][1] + 1) + " " + std::to_string(faces[i][2] + 1) + "\n";
	}
	out << "\n";
	out.close();
	return prefix;
}
std::pair<std::string, std::string> produceDiffuseLight(std::string outfileMtl, minipbrt::DiffuseAreaLight* toExport, int number) {
	Eigen::Vector3f scale = Eigen::Vector3f(toExport->scale);
	Eigen::Vector3f intensity = Eigen::Vector3f(toExport->L);
	std::string expString = "mdl 1.7;\n";
	expString += "import ::df::*;\n";
	expString += "\nexport material " + outfileMtl + "_" + "diffuse_light" + std::to_string(number) + "(\n";
	expString += "	uniform color parIntensityTint = color(" + std::to_string(intensity[0] * scale[0] * intensity[0] * scale[0]) + ", " + std::to_string(intensity[1] * scale[1] * intensity[1] * scale[1]) + ", " + std::to_string(intensity[2] * scale[2] * intensity[2] * scale[2]) + "),\n";
	expString += "	uniform float parIntensity = 1";
	expString += ")\n=\n";
	expString += "	material(\n";
	expString += "		surface: material_surface(\n";
	expString += "			emission : material_emission(\n";
	expString += "				emission : df::diffuse_edf(),\n";
	expString += "              intensity : parIntensityTint * parIntensity,\n";
	expString += "              mode : intensity_radiant_exitance\n";
	expString += "          )\n";
	expString += "      )\n";
	expString += "  );";
	std::ofstream out(outfileMtl + "_" + std::to_string(number) + "_" + "diffuse_light.mdl");
	out << expString;
	out.close();
	return std::make_pair(outfileMtl + "_" + std::to_string(number) + "_" + "diffuse_light.mdl", outfileMtl + "_" + "diffuse_light" + std::to_string(number));
}
int main() {
	minipbrt::Loader loader;
	std::string outfileMtl = "bmw";
	std::vector<std::pair<std::string, uint32_t>> objToMaterial;
	std::vector<std::vector<Eigen::Vector3f>> SRT;
	std::vector<uint32_t> isLight;
	if (loader.load("bmw-m6.pbrt")) {
		minipbrt::Scene* scene = loader.take_scene();
		scene->load_all_ply_meshes();
		int ind = 0;
		// All objs are already present and can be poached from the pbrt
		for (auto itr = scene->shapes.begin(); itr != scene->shapes.end(); itr++) {
			minipbrt::Transform ctm = (*itr)->shapeToWorld;
			Eigen::Vector3f translation = Eigen::Vector3f(ctm.start[0][3], ctm.start[1][3], ctm.start[2][3]);
			// Given that a CTM with no scale factor should be orthnormal, take columns and derive scale from them.
			Eigen::Vector3f rowA = { ctm.start[0][0],ctm.start[0][1],ctm.start[0][2] };
			Eigen::Vector3f rowB = { ctm.start[1][0],ctm.start[1][1],ctm.start[1][2] };
			Eigen::Vector3f rowC = { ctm.start[2][0],ctm.start[2][1],ctm.start[2][2] };
			Eigen::Vector3f scalars = Eigen::Vector3f(rowA.norm(), rowB.norm(), rowC.norm());
			Eigen::Matrix3f rot { 
				{ ctm.start[0][0] / scalars[0] ,ctm.start[0][1] / scalars[0], ctm.start[0][2] / scalars[0]},
				{ ctm.start[1][0] / scalars[1] ,ctm.start[1][1] / scalars[1], ctm.start[1][2] / scalars[1]},
				{ ctm.start[2][0] / scalars[2] ,ctm.start[2][1] / scalars[2], ctm.start[2][2] / scalars[2]} };
			// Derive scale factors from columns. (Note, has to be done manually to avoid divbyzero)
			// Indexing may also be wrong, check back later.
			float xRot = 180.f / 3.14 * atan2(rot(2, 1), rot(2, 2));
			float yRot = 180.f / 3.14 * atan2(-rot(2, 0), sqrt(rot(2, 1) * rot(2, 1) + rot(2, 2) * rot(2, 2)));
			float zRot = 180.f / 3.14 * atan2(rot(1, 0), rot(0, 0));
			// Next, handle shapes. ATM, some shapes cause more issues because I need to manually map to mtl
			// But this should handle the basics for now.

			if ((*itr)->type() == minipbrt::ShapeType::TriangleMesh) {
				minipbrt::TriangleMesh* toExport = static_cast<minipbrt::TriangleMesh*>(*itr);
				std::string objFile = produceObjFromTriMesh(toExport, outfileMtl + "_" + std::to_string(ind) + ".obj");
				ind += 1;
				objToMaterial.push_back(std::make_pair(objFile, toExport->material));
				SRT.push_back(std::vector<Eigen::Vector3f>());
				SRT[SRT.size() - 1].push_back(scalars);
				SRT[SRT.size() - 1].push_back(Eigen::Vector3f(xRot, yRot, zRot));
				SRT[SRT.size() - 1].push_back(translation);
				if ((*itr)->areaLight != 4294967295) {
					isLight.push_back((*itr)->areaLight);
				}
				else {
					isLight.push_back(4294967294);
				}
			}

		}
		// ... process the scene, then delete it ...
		// process shape materials and store for obj_material referrencing later.
		std::vector<std::string> idToMaterialFiles;
		std::vector<std::string> idToMaterialName;
		std::vector<std::string> textures;
		for (int i = 0; i < scene->textures.size(); i++) {
			minipbrt::TextureType texType = scene->textures[i]->type();
			if (texType == minipbrt::TextureType::ImageMap) {
				minipbrt::ImageMapTexture* toExport = static_cast<minipbrt::ImageMapTexture*>(scene->textures[i]);
				textures.push_back(toExport->filename);
			}
			else {
				textures.push_back("none");
			}
		}
		for (int i = 0; i < scene->materials.size(); i++) {
			// get components for specular/
			minipbrt::MaterialType matType = scene->materials[i]->type();
			Eigen::Vector3f kD = Eigen::Vector3f(0,0,0);
			Eigen::Vector3f kS = Eigen::Vector3f(0,0,0);
			Eigen::Vector3f kT = Eigen::Vector3f(0,0,0);
			float roughness = 0;
			int contains = 0;
			// TODO IF NECESSARY. HANDLE TEXTURE EXPORT. FOR FRIDAY THIS SHOULD NOT BE NECESSARY.
			std::pair<std::string, std::string> fileName;
			if (matType == minipbrt::MaterialType::Plastic) {
				// TODO 
				minipbrt::PlasticMaterial* toExport = static_cast<minipbrt::PlasticMaterial*>(scene->materials[i]);
				fileName = producePlasticMDL(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Matte) {
				minipbrt::MatteMaterial* toExport = static_cast<minipbrt::MatteMaterial*>(scene->materials[i]);
				fileName = produceMatteMDL(outfileMtl, toExport, i, textures);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Mirror) {
				// Special case of specualr reflection
				minipbrt::MirrorMaterial* toExport = static_cast<minipbrt::MirrorMaterial*>(scene->materials[i]);
				fileName = produceMirrorMDL(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Metal) {
				minipbrt::MetalMaterial* toExport = static_cast<minipbrt::MetalMaterial*>(scene->materials[i]);
				fileName = produceMetalMDL(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Glass) {
				minipbrt::GlassMaterial* toExport = static_cast<minipbrt::GlassMaterial*>(scene->materials[i]);
				fileName = produceGlassMDL(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Uber) {
				// this material sucks to deal with. For now treating this way
				minipbrt::UberMaterial* toExport = static_cast<minipbrt::UberMaterial*>(scene->materials[i]);
				fileName = produceUberMDL(outfileMtl, toExport, i, textures);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Substrate) {
				minipbrt::SubstrateMaterial* toExport = static_cast<minipbrt::SubstrateMaterial*>(scene->materials[i]);
				fileName = produceSubstrateMDL(outfileMtl, toExport, i, textures);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
			}
			else if (matType == minipbrt::MaterialType::Fourier){
				minipbrt::FourierMaterial* toExport = static_cast<minipbrt::FourierMaterial*>(scene->materials[i]);
				fileName = produceFourierMDL(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
				std::cout << "WARNING. FOURIER MATERIALS NOT PARSEABLE. PRODUCING DEFAULT METAL" << std::endl;
			}
			else if (matType == minipbrt::MaterialType::Mix) {
				// Currently not handling mixed materials... This would involve handling the fourier case.
				minipbrt::MixMaterial* toExport = static_cast<minipbrt::MixMaterial*>(scene->materials[i]);
				fileName = produceMixed(outfileMtl, toExport, i);
				idToMaterialFiles.push_back(fileName.first);
				idToMaterialName.push_back(fileName.second);
				std::cout << "WARNING. MIXED MATERIALS NOT PARSEABLE DUE TO FOURIER SUB COMPONENT. PRODUCING DEFAULT METAL" << std::endl;
			}
			else if (matType == minipbrt::MaterialType::None){
				idToMaterialFiles.push_back("none");
				idToMaterialName.push_back("none");
				std::cout << "WARNING. NONE MATERIAL! None materials are non-valid in mdl syntax, and have been ignored" << std::endl;
			} else {
				auto dbg = 3;
			}
			// TODO. Assign variable allocations for different files.
		}
		std::vector<std::string> idToLightFiles;
		std::vector<std::string> idToLightNames;
		for (int i = 0; i < scene->areaLights.size(); i++) {
			if (scene->areaLights[i]->type() == minipbrt::AreaLightType::Diffuse) {
				minipbrt::DiffuseAreaLight* toExport = static_cast<minipbrt::DiffuseAreaLight*>(scene->areaLights[i]);
				std::pair<std::string, std::string> fileName = produceDiffuseLight(outfileMtl, toExport, i);
				idToLightFiles.push_back(fileName.first);
				idToLightNames.push_back(fileName.second);
			}
			else {
				idToLightFiles.push_back("none");
				idToLightNames.push_back("none");
			}
		}
		// Given that all meshed objects have been dumped and we have what we need for object->material associations, celebrate.
		// Begin creating file
		std::ofstream out("scene_" + outfileMtl + ".txt");
		// Apply camera transforms:
		minipbrt::Transform ctm = scene->camera->cameraToWorld;
		Eigen::Vector3f translation = Eigen::Vector3f(ctm.start[0][3], ctm.start[1][3], ctm.start[2][3]);
		// Next, handle shapes. ATM, some shapes cause more issues because I need to manually map to mtl
		out << "lensShader 0\n";
		out << "center 0 0 0\n";
		out << "camera " + std::to_string(translation[0]) + " " + std::to_string(translation[1]) + " " + std::to_string(translation[2]) + "\n";
		out << "gamma 2.2\n";
		out << "colorBalance 1 1 1\n";
		out << "whitePoint 1\n";
		out << "burnHighlights 0.8\n";
		out << "crushBlacks 0.2\n";
		out << "saturation 1.2\n";
		out << "brightness 1\n";
		out << "\n";
		out << "# ========== MATERIALS\n\n";
		std::vector<std::string> materialNames;
		std::vector<std::string> lightNames;
		out << "mdl default " + idToMaterialName[0] + " \"" + idToMaterialFiles[0] + "\"\n\n";
		for (int i = 0; i < idToMaterialFiles.size(); i++) {
			if (idToMaterialFiles[i] != "none") {
				materialNames.push_back("material_" + std::to_string(i) + "_bsdf ");
				out << "mdl material_" + std::to_string(i) + "_bsdf " + idToMaterialName[i] + " \"" + idToMaterialFiles[i] + "\"\n\n";
			}
			else {
				materialNames.push_back("none");
			}
		}
		for (int i = 0; i < idToLightFiles.size(); i++) {
			if (idToLightFiles[i] != "none") {
				lightNames.push_back("material_" + std::to_string(i) + "_bsdf ");
				out << "mdl material_" + std::to_string(i) + "_light " + idToLightNames[i] + " \"" + idToLightFiles[i] + "\"\n\n";
			}
			else {
				materialNames.push_back("none");
			}
		}
		if (scene->areaLights.size() == 0) {
			out << "# ========== LIGHTS\n\n";
			out << "push\n";
			out << "emission 1 1 1\n";
			out << "emissionMultiplier 1\n";
			out << "rotate 0 1 0 180\n";
			out << "light env\n";
			out << "pop\n";
			out << "push\n";
		}
		out << "# ========== GEOMETRY\n\n";
		for (int i = 0; i < SRT.size(); i++) {
			if (isLight[i] == 4294967294) {
					if (objToMaterial[i].second < materialNames.size() && materialNames[objToMaterial[i].second] != "none") {
						out << "push\n";
						out << "scale " + std::to_string(SRT[i][0][0]) + " " + std::to_string(SRT[i][0][1]) + " " + std::to_string(SRT[i][0][2]) + "\n";
						out << "rotate 1 0 0 " + std::to_string(SRT[i][1][0]) + "\n";
						out << "rotate 0 1 0 " + std::to_string(SRT[i][1][1]) + "\n";
						out << "rotate 0 0 1 " + std::to_string(SRT[i][1][2]) + "\n";
						out << "translate " + std::to_string(SRT[i][2][0]) + " " + std::to_string(SRT[i][2][1]) + " " + std::to_string(SRT[i][2][2]) + "\n";
						out << "model assimp \"" + objToMaterial[i].first + "\" " + materialNames[objToMaterial[i].second] + "\n";
						out << "pop\n\n";
					}
			}
			else {
				if (idToLightNames[isLight[i]] != "none") {
					out << "push\n";
					out << "scale " + std::to_string(SRT[i][0][0]) + " " + std::to_string(SRT[i][0][1]) + " " + std::to_string(SRT[i][0][2]) + "\n";
					out << "rotate 1 0 0 " + std::to_string(SRT[i][1][0]) + "\n";
					out << "rotate 0 1 0 " + std::to_string(SRT[i][1][1]) + "\n";
					out << "rotate 0 0 1 " + std::to_string(SRT[i][1][2]) + "\n";
					out << "translate " + std::to_string(SRT[i][2][0]) + " " + std::to_string(SRT[i][2][1]) + " " + std::to_string(SRT[i][2][2]) + "\n";
					out << "model assimp \"" + objToMaterial[i].first + "\" material_" + std::to_string(isLight[i]) + "_light\n";
					out << "pop\n\n";
				}
			}
		}
		out.close();
		delete scene;
	}
}