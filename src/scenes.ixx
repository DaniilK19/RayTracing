// Module scenes: scenes, i.e. what to render (objects) and how to shoot it (camera settings).
export module scenes;

import std;
import camera;
import hittable;
import image;
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
            return std::filesystem::path("assets") / name; // reached the drive root: the loading code will report the failure
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
};

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
            }};
}

// All scenes, in the order they appear in the books.
export constexpr std::array scene_list{
    scene_entry{"weekend", "the first book's cover: spheres of three materials", weekend},
    scene_entry{"bouncing", "the same, but the matte spheres bounce: motion blur", bouncing},
    scene_entry{"checker", "two spheres with a solid checker texture", checker},
    scene_entry{"earth", "the Earth: an image wrapped around a sphere", earth},
    scene_entry{"perlin", "marble spheres made of Perlin noise", perlin_spheres},
    scene_entry{"quads", "five colored quadrilaterals", quads},
};

export constexpr std::string_view default_scene = "weekend";

// The scene with the given name, or nullptr if there is none. The projection &scene_entry::name (C++20):
// std::ranges::find compares name not with the elements themselves but with their name field.
export const scene_entry* find_scene(std::string_view name)
{
    const auto it = std::ranges::find(scene_list, name, &scene_entry::name);
    return it != scene_list.end() ? &*it : nullptr;
}
