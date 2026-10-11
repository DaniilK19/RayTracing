// Module scenes: scenes, i.e. what to render (objects) and how to shoot it (camera settings).
export module scenes;

import std;
import bvh;
import camera;
import constant_medium;
import hittable;
import image;
import instance;
import material;
import quad;
import random;
import sphere;
import texture;
import vec3;

// A scene: objects and the camera that shoots them.
export struct scene
{
    hittable_list world;
    camera_settings camera;

    // What the lights and mixture strategies aim at (book 3, chapter 9): a copy of each lamp's shape (or of another
    // object worth aiming at, see glass), without a material, as only its pdf_value and random are used. If the copy
    // drifted away from the original, rays would be aimed where no light comes from, with no error anywhere. So each
    // scene builds both from one function that takes the material: the real one for world, nullptr for lights.
    // It comes last, so scenes lit by the sky can leave it out: return {std::move(world), camera}.
    hittable_list lights;
};

// A scene in the list: a name for the command line, a description for the help and the function that builds it.
// A function pointer is stored rather than a ready scene: only the chosen scene gets built.
export struct scene_entry
{
    std::string_view name;
    std::string_view description;
    scene (*build)();
};

// A file from the project's assets folder. We look for the folder in the current one and up the tree:
// the program is run both from the project root (out\build\x64-release\RayTracing.exe) and from Visual Studio,
// where the current folder is out\build\x64-release and assets lies three levels up.
std::filesystem::path find_asset(std::string_view name)
{
    for (std::filesystem::path dir = std::filesystem::current_path();; dir = dir.parent_path())
    {
        const std::filesystem::path candidate = dir / "assets" / name;
        if (std::filesystem::exists(candidate))
        {
            return candidate;
        }
        if (dir == dir.parent_path())
        {
            // Reached the drive root: the loading code will report the failure
            return std::filesystem::path("assets") / name;
        }
    }
}

// A texture from an image in assets. If the file doesn't load, rendering still happens:
// the image is replaced with bright cyan, which immediately shows that the texture is missing.
std::shared_ptr<texture> load_texture(std::string_view name)
{
    auto img = image::load(find_asset(name));
    if (!img)
    {
        std::println("Texture {}: {}. Using cyan instead.", name, img.error());
        return std::make_shared<solid_color>(color{0, 1, 1});
    }
    return std::make_shared<image_texture>(std::move(*img));
}

// The first book's cover: three big spheres and a scatter of small ones with random materials.
// The main thread's random number generator always starts from the same seed, so the scene
// is the same on every run.
// bouncing is the second book's variant: the matte spheres bounce while the shutter is open,
// and the floor became a checkerboard.
hittable_list random_spheres(bool bouncing)
{
    hittable_list world;

    std::shared_ptr<texture> ground = std::make_shared<solid_color>(color{0.5, 0.5, 0.5});
    if (bouncing)
    {
        ground = std::make_shared<checker_texture>(0.32, color{0.2, 0.3, 0.1}, color{0.9, 0.9, 0.9});
    }
    world.add(std::make_unique<sphere>(point3{0, -1000, 0}, 1000, std::make_shared<lambertian>(std::move(ground))));

    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            const double choose_mat = random_double();
            const point3 center{a + 0.9 * random_double(), 0.2, b + 0.9 * random_double()};
            if ((center - point3{4, 0.2, 0}).length() <= 0.9)
            {
                continue; // don't put a small sphere right next to the big metal one
            }

            std::shared_ptr<material> sphere_material;
            point3 center2 = center; // where the sphere will be by the end of the exposure
            if (choose_mat < 0.8)    // 80% matte
            {
                sphere_material = std::make_shared<lambertian>(color::random() * color::random());
                if (bouncing)
                {
                    center2 = center + vec3{0, random_double(0, 0.5), 0}; // jumps up by a random height of up to 0.5
                }
            }
            else if (choose_mat < 0.95) // 15% metal
            {
                sphere_material = std::make_shared<metal>(color::random(0.5, 1), random_double(0, 0.5));
            }
            else // 5% glass
            {
                sphere_material = std::make_shared<dielectric>(1.5);
            }
            world.add(std::make_unique<sphere>(center, center2, 0.2, std::move(sphere_material)));
        }
    }

    world.add(std::make_unique<sphere>(point3{0, 1, 0}, 1.0, std::make_shared<dielectric>(1.5)));
    world.add(std::make_unique<sphere>(point3{-4, 1, 0}, 1.0, std::make_shared<lambertian>(color{0.4, 0.2, 0.1})));
    world.add(std::make_unique<sphere>(point3{4, 1, 0}, 1.0, std::make_shared<metal>(color{0.7, 0.6, 0.5}, 0.0)));
    return world;
}

// Camera for both covers
constexpr camera_settings cover_camera{
    .aspect_ratio = 16.0 / 9.0,
    .image_width = 1920, // Full HD: with a 16:9 aspect ratio the height will be 1080
    .samples_per_pixel = 500,
    .max_depth = 50,
    .vfov = 20,
    .lookfrom = {13, 2, 3},
    .lookat = {0, 0, 0},
    .vup = {0, 1, 0},
    .defocus_angle = 0.6,
    .focus_dist = 10.0,
    .background = daylight_sky,
};

// The first book's chapters 9–13 in one picture, as chapter 13 leaves it: a matte sphere in the middle, a hollow glass
// sphere on the left, a very fuzzy metal one on the right, all standing on a big matte sphere, the ground. The glass is
// hollow because a smaller sphere inside it has the refractive index of air relative to glass, 1 / 1.5. The lens is
// wide open (a 10° cone of rays through each pixel), so only what is 3.4 units from the camera comes out sharp.
scene materials()
{
    hittable_list world;
    const auto material_ground = std::make_shared<lambertian>(color{0.8, 0.8, 0.0});
    const auto material_center = std::make_shared<lambertian>(color{0.1, 0.2, 0.5});
    const auto material_left = std::make_shared<dielectric>(1.50);
    const auto material_bubble = std::make_shared<dielectric>(1.00 / 1.50);
    const auto material_right = std::make_shared<metal>(color{0.8, 0.6, 0.2}, 1.0);

    world.add(std::make_unique<sphere>(point3{0.0, -100.5, -1.0}, 100.0, material_ground));
    world.add(std::make_unique<sphere>(point3{0.0, 0.0, -1.2}, 0.5, material_center));
    world.add(std::make_unique<sphere>(point3{-1.0, 0.0, -1.0}, 0.5, material_left));
    world.add(std::make_unique<sphere>(point3{-1.0, 0.0, -1.0}, 0.4, material_bubble)); // the air inside the glass
    world.add(std::make_unique<sphere>(point3{1.0, 0.0, -1.0}, 0.5, material_right));

    camera_settings camera = cover_camera;
    camera.lookfrom = {-2, 2, 1};
    camera.lookat = {0, 0, -1};
    camera.defocus_angle = 10.0;
    camera.focus_dist = 3.4;
    return {std::move(world), camera};
}

scene weekend()
{
    return {random_spheres(false), cover_camera};
}

// Book 2, chapter 2: motion blur
scene bouncing()
{
    return {random_spheres(true), cover_camera};
}

// Book 2, chapter 4: two huge spheres with a solid checker texture. The squares on them are slices
// of the same cubes of space.
scene checker()
{
    hittable_list world;
    const auto checker = std::make_shared<lambertian>(
        std::make_shared<checker_texture>(0.32, color{0.2, 0.3, 0.1}, color{0.9, 0.9, 0.9}));
    world.add(std::make_unique<sphere>(point3{0, -10, 0}, 10, checker));
    world.add(std::make_unique<sphere>(point3{0, 10, 0}, 10, checker));

    camera_settings camera = cover_camera;
    camera.defocus_angle = 0; // no defocus: everything is sharp
    return {std::move(world), camera};
}

// Book 2, chapter 4: the Earth, an image wrapped around a sphere.
// assets/earthmap.jpg comes from the book's repository (github.com/RayTracing/raytracing.github.io, CC0 license).
scene earth()
{
    hittable_list world;
    const auto earth_surface = std::make_shared<lambertian>(load_texture("earthmap.jpg"));
    world.add(std::make_unique<sphere>(point3{0, 0, 0}, 2, earth_surface));

    camera_settings camera = cover_camera;
    camera.lookfrom = {0, 0, 12};
    camera.defocus_angle = 0;
    return {std::move(world), camera};
}

// Book 2, chapter 5: marble spheres, a texture made of Perlin noise.
scene perlin_spheres()
{
    hittable_list world;
    const auto marble = std::make_shared<lambertian>(std::make_shared<noise_texture>(4));
    world.add(std::make_unique<sphere>(point3{0, -1000, 0}, 1000, marble));
    world.add(std::make_unique<sphere>(point3{0, 2, 0}, 2, marble));

    camera_settings camera = cover_camera;
    camera.defocus_angle = 0;
    return {std::move(world), camera};
}

// Book 2, chapter 6: five colored quadrilaterals around the camera.
scene quads()
{
    hittable_list world;
    const auto left_red = std::make_shared<lambertian>(color{1.0, 0.2, 0.2});
    const auto back_green = std::make_shared<lambertian>(color{0.2, 1.0, 0.2});
    const auto right_blue = std::make_shared<lambertian>(color{0.2, 0.2, 1.0});
    const auto upper_orange = std::make_shared<lambertian>(color{1.0, 0.5, 0.0});
    const auto lower_teal = std::make_shared<lambertian>(color{0.2, 0.8, 0.8});

    world.add(std::make_unique<quad>(point3{-3, -2, 5}, vec3{0, 0, -4}, vec3{0, 4, 0}, left_red));
    world.add(std::make_unique<quad>(point3{-2, -2, 0}, vec3{4, 0, 0}, vec3{0, 4, 0}, back_green));
    world.add(std::make_unique<quad>(point3{3, -2, 1}, vec3{0, 0, 4}, vec3{0, 4, 0}, right_blue));
    world.add(std::make_unique<quad>(point3{-2, 3, 1}, vec3{4, 0, 0}, vec3{0, 0, 4}, upper_orange));
    world.add(std::make_unique<quad>(point3{-2, -3, 5}, vec3{4, 0, 0}, vec3{0, 0, -4}, lower_teal));

    return {std::move(world),
            {
                .aspect_ratio = 1.0,
                .image_width = 1080, // square scenes are as tall as Full HD
                .samples_per_pixel = 500,
                .max_depth = 50,
                .vfov = 80,
                .lookfrom = {0, 0, 9},
                .lookat = {0, 0, 0},
                .vup = {0, 1, 0},
                .background = daylight_sky,
            }};
}

// Book 2, chapter 7: marble spheres in the dark, lit by a glowing sphere and a glowing rectangle.
scene simple_light()
{
    hittable_list world;
    const auto marble = std::make_shared<lambertian>(std::make_shared<noise_texture>(4));
    world.add(std::make_unique<sphere>(point3{0, -1000, 0}, 1000, marble));
    world.add(std::make_unique<sphere>(point3{0, 2, 0}, 2, marble));

    // Lambdas build the two lamps for world and for lights (see scene::lights). The quad lamp faces +z.
    const auto sphere_lamp = [](std::shared_ptr<const material> mat) {
        return std::make_unique<sphere>(point3{0, 7, 0}, 2, std::move(mat));
    };
    const auto quad_lamp = [](std::shared_ptr<const material> mat) {
        return std::make_unique<quad>(point3{3, 1, -2}, vec3{2, 0, 0}, vec3{0, 2, 0}, std::move(mat));
    };
    const auto light = std::make_shared<diffuse_light>(color{4, 4, 4});
    world.add(sphere_lamp(light));
    world.add(quad_lamp(light));

    hittable_list lights;
    lights.add(quad_lamp(nullptr));
    lights.add(sphere_lamp(nullptr)); // a sphere can be aimed at since book 3, sections 12.3–12.4

    return {std::move(world),
            {
                .aspect_ratio = 16.0 / 9.0,
                .image_width = 1920,
                .samples_per_pixel = 1000, // darkness with small lamps means more noise
                .max_depth = 50,
                .vfov = 20,
                .lookfrom = {26, 3, 6},
                .lookat = {0, 2, 0},
                .vup = {0, 1, 0},
            }, // the background is black by default
            std::move(lights)};
}

// The Cornell box is a classic scene for testing lighting (Cornell University, 1984):
// a 555×555×555 room with red and green walls and a lamp on the ceiling. There is no front wall: the camera is there.
// Adds five walls to world; each scene places its own lamp.
void add_cornell_walls(hittable_list& world)
{
    const auto red = std::make_shared<lambertian>(color{0.65, 0.05, 0.05});
    const auto white = std::make_shared<lambertian>(color{0.73, 0.73, 0.73});
    const auto green = std::make_shared<lambertian>(color{0.12, 0.45, 0.15});

    world.add(std::make_unique<quad>(point3{555, 0, 0}, vec3{0, 555, 0}, vec3{0, 0, 555}, green));      // left
    world.add(std::make_unique<quad>(point3{0, 0, 0}, vec3{0, 555, 0}, vec3{0, 0, 555}, red));          // right
    world.add(std::make_unique<quad>(point3{0, 0, 0}, vec3{555, 0, 0}, vec3{0, 0, 555}, white));        // floor
    world.add(std::make_unique<quad>(point3{555, 555, 555}, vec3{-555, 0, 0}, vec3{0, 0, -555}, white)); // ceiling
    world.add(std::make_unique<quad>(point3{0, 0, 555}, vec3{555, 0, 0}, vec3{0, 555, 0}, white));      // back
}

// The tall and short blocks of the Cornell box: a box at the origin, rotated around
// the vertical and moved into place. The order matters: rotation is around the origin,
// so we rotate first and translate second.
std::unique_ptr<hittable> tall_block(std::shared_ptr<const material> mat)
{
    std::unique_ptr<hittable> block = box(point3{0, 0, 0}, point3{165, 330, 165}, std::move(mat));
    block = std::make_unique<rotate_y>(std::move(block), 15);
    return std::make_unique<translate>(std::move(block), vec3{265, 0, 295});
}

std::unique_ptr<hittable> short_block(std::shared_ptr<const material> mat)
{
    std::unique_ptr<hittable> block = box(point3{0, 0, 0}, point3{165, 165, 165}, std::move(mat));
    block = std::make_unique<rotate_y>(std::move(block), -18);
    return std::make_unique<translate>(std::move(block), vec3{130, 0, 65});
}

// The lamp of the Cornell box: 130×105, 1 unit below the ceiling, facing down. Built for world and for the
// scene's lights (see scene::lights).
std::unique_ptr<hittable> cornell_lamp(std::shared_ptr<const material> mat)
{
    return std::make_unique<quad>(point3{343, 554, 332}, vec3{-130, 0, 0}, vec3{0, 0, -105}, std::move(mat));
}

constexpr camera_settings cornell_camera{
    .aspect_ratio = 1.0,
    .image_width = 1080,
    .samples_per_pixel = 1000, // a small lamp in a closed room: noisy unless rays are aimed at it (the default)
    .max_depth = 50,
    .vfov = 40,
    .lookfrom = {278, 278, -800},
    .lookat = {278, 278, 0},
    .vup = {0, 1, 0},
};

// Book 2, chapters 7–8: the Cornell box with two blocks
scene cornell()
{
    hittable_list world;
    add_cornell_walls(world);
    world.add(cornell_lamp(std::make_shared<diffuse_light>(color{15, 15, 15})));

    const auto white = std::make_shared<lambertian>(color{0.73, 0.73, 0.73});
    world.add(tall_block(white));
    world.add(short_block(white));

    hittable_list lights;
    lights.add(cornell_lamp(nullptr));
    return {std::move(world), cornell_camera, std::move(lights)};
}

// Book 2, chapter 9: the same box, but the blocks are made of smoke, black and white. The lamp is bigger and dimmer.
scene smoke()
{
    hittable_list world;
    add_cornell_walls(world);
    // The lamp faces down. A lambda builds it for world and for lights (see scene::lights).
    const auto lamp = [](std::shared_ptr<const material> mat) {
        return std::make_unique<quad>(point3{113, 554, 127}, vec3{330, 0, 0}, vec3{0, 0, 305}, std::move(mat));
    };
    world.add(lamp(std::make_shared<diffuse_light>(color{7, 7, 7})));

    // The material of the block's walls doesn't matter: only the shape of the volume is used
    const auto white = std::make_shared<lambertian>(color{0.73, 0.73, 0.73});
    world.add(std::make_unique<constant_medium>(tall_block(white), 0.01, color{0, 0, 0}));
    world.add(std::make_unique<constant_medium>(short_block(white), 0.01, color{1, 1, 1}));

    hittable_list lights;
    lights.add(lamp(nullptr));
    return {std::move(world), cornell_camera, std::move(lights)};
}

// Book 2, chapter 10: the final scene, everything from the second book together.
scene final_scene()
{
    hittable_list world;

    // The floor is a field of 20×20 boxes of random height. They have their own BVH, which becomes a single
    // object of the overall tree built by main.
    hittable_list ground_boxes;
    const auto ground = std::make_shared<lambertian>(color{0.48, 0.83, 0.53});
    constexpr int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; ++i)
    {
        for (int j = 0; j < boxes_per_side; ++j)
        {
            constexpr double w = 100.0;
            const point3 corner0{-1000.0 + i * w, 0.0, -1000.0 + j * w};
            const point3 corner1{corner0.x + w, random_double(1, 101), corner0.z + w};
            ground_boxes.add(box(corner0, corner1, ground));
        }
    }
    world.add(build_bvh(std::move(ground_boxes)));

    // The lamp faces down. A lambda builds it for world and for lights (see scene::lights).
    const auto lamp = [](std::shared_ptr<const material> mat) {
        return std::make_unique<quad>(point3{123, 554, 147}, vec3{300, 0, 0}, vec3{0, 0, 265}, std::move(mat));
    };
    world.add(lamp(std::make_shared<diffuse_light>(color{7, 7, 7})));

    // A moving sphere: motion blur
    const point3 center1{400, 400, 200};
    const point3 center2 = center1 + vec3{30, 0, 0};
    world.add(std::make_unique<sphere>(center1, center2, 50, std::make_shared<lambertian>(color{0.7, 0.3, 0.1})));

    // Glass and matte-metal spheres
    world.add(std::make_unique<sphere>(point3{260, 150, 45}, 50, std::make_shared<dielectric>(1.5)));
    world.add(std::make_unique<sphere>(point3{0, 150, 145}, 50, std::make_shared<metal>(color{0.8, 0.8, 0.9}, 1.0)));

    // A glass sphere filled with blue fog: the glass and the fog are two objects of the same shape.
    // In the book it is one and the same shared_ptr; here each object has a single owner, hence two spheres.
    const auto glass = std::make_shared<dielectric>(1.5);
    world.add(std::make_unique<sphere>(point3{360, 150, 145}, 70, glass));
    world.add(std::make_unique<constant_medium>(std::make_unique<sphere>(point3{360, 150, 145}, 70, glass), 0.2,
                                                color{0.2, 0.4, 0.9}));

    // A light haze over the whole scene: a huge sphere of very thin fog with the camera inside it
    world.add(std::make_unique<constant_medium>(std::make_unique<sphere>(point3{0, 0, 0}, 5000, glass), 0.0001,
                                                color{1, 1, 1}));

    // The Earth and a marble sphere
    world.add(std::make_unique<sphere>(point3{400, 200, 400}, 100,
                                       std::make_shared<lambertian>(load_texture("earthmap.jpg"))));
    world.add(std::make_unique<sphere>(point3{220, 280, 300}, 80,
                                       std::make_shared<lambertian>(std::make_shared<noise_texture>(0.2))));

    // A cloud of 1000 white balls in a 165×165×165 cube, also with its own BVH, rotated and translated
    hittable_list cloud;
    const auto white = std::make_shared<lambertian>(color{0.73, 0.73, 0.73});
    for (int i = 0; i < 1000; ++i)
    {
        cloud.add(std::make_unique<sphere>(vec3::random(0, 165), 10, white));
    }
    world.add(std::make_unique<translate>(std::make_unique<rotate_y>(build_bvh(std::move(cloud)), 15),
                                          vec3{-100, 270, 395}));

    hittable_list lights;
    lights.add(lamp(nullptr));

    return {std::move(world),
            {
                .aspect_ratio = 1.0,
                .image_width = 1080,
                .samples_per_pixel = 1000, // ~4 minutes on all cores; the book uses 10,000 at 800×800
                .max_depth = 40,
                .vfov = 40,
                .lookfrom = {478, 278, -600},
                .lookat = {278, 278, 0},
                .vup = {0, 1, 0},
            },
            std::move(lights)};
}

// Book 3, section 12.2: the Cornell box, but the tall block is polished aluminum, a perfect mirror (fuzz 0). The
// camera follows a mirror's ray as before (the law of reflection fixes its direction), and lamp sampling aims at the
// lamp itself, not at its image in the mirror. So the lamp's light that the block's top reflects onto the ceiling is
// found only by the ceiling's samples that happen to hit the mirror: that bright patch stays noisy, as in the book.
scene aluminum()
{
    hittable_list world;
    add_cornell_walls(world);
    world.add(cornell_lamp(std::make_shared<diffuse_light>(color{15, 15, 15})));

    world.add(tall_block(std::make_shared<metal>(color{0.8, 0.85, 0.88}, 0.0)));
    world.add(short_block(std::make_shared<lambertian>(color{0.73, 0.73, 0.73})));

    hittable_list lights;
    lights.add(cornell_lamp(nullptr));
    return {std::move(world), cornell_camera, std::move(lights)};
}

// Book 3, sections 12.3–12.5, the book's last image: the Cornell box with a glass sphere instead of the short block.
// The sphere gathers the lamp's light into a bright spot on the floor, a caustic. It is in the list of lights too,
// although it doesn't glow: the list is what the mixture strategy aims at. Lamp light that reaches a surface by way
// of the sphere is found only by rays that happen to enter it; with a quarter of the rays aimed at it, rare bright
// dots (fireflies) turn into even grain. The price: rays aimed at the lamp end there, but those aimed at the sphere
// go on through it, so a sample takes about twice as long.
scene glass()
{
    hittable_list world;
    add_cornell_walls(world);
    world.add(cornell_lamp(std::make_shared<diffuse_light>(color{15, 15, 15})));
    world.add(tall_block(std::make_shared<lambertian>(color{0.73, 0.73, 0.73})));

    // A lambda builds the sphere for world and for lights (see scene::lights)
    const auto ball = [](std::shared_ptr<const material> mat) {
        return std::make_unique<sphere>(point3{190, 90, 190}, 90, std::move(mat));
    };
    world.add(ball(std::make_shared<dielectric>(1.5)));

    hittable_list lights;
    lights.add(cornell_lamp(nullptr));
    lights.add(ball(nullptr));
    return {std::move(world), cornell_camera, std::move(lights)};
}

// All scenes, in the order they appear in the books.
export constexpr std::array scene_list{
    scene_entry{"materials", "the first book's three materials up close: matte, fuzzy metal, hollow glass", materials},
    scene_entry{"weekend", "the first book's cover: spheres of three materials", weekend},
    scene_entry{"bouncing", "the same, but the matte spheres bounce: motion blur", bouncing},
    scene_entry{"checker", "two spheres with a solid checker texture", checker},
    scene_entry{"earth", "the Earth: an image wrapped around a sphere", earth},
    scene_entry{"perlin", "marble spheres made of Perlin noise", perlin_spheres},
    scene_entry{"quads", "five colored quadrilaterals", quads},
    scene_entry{"light", "marble spheres in the dark and two light sources", simple_light},
    scene_entry{"cornell", "the Cornell box, a classic scene for testing lighting", cornell},
    scene_entry{"smoke", "the Cornell box with blocks of black and white smoke", smoke},
    scene_entry{"final", "the second book's final scene: everything together", final_scene},
    scene_entry{"aluminum", "the Cornell box with a tall block of polished aluminum", aluminum},
    scene_entry{"glass", "the Cornell box with a glass sphere: the third book's last image", glass},
};

export constexpr std::string_view default_scene = "final";

// The scene with the given name, or nullptr if there is none. The projection &scene_entry::name (C++20):
// std::ranges::find compares name not with the elements themselves but with their name field.
export const scene_entry* find_scene(std::string_view name)
{
    const auto it = std::ranges::find(scene_list, name, &scene_entry::name);
    return it != scene_list.end() ? &*it : nullptr;
}
