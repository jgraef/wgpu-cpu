use crate::render_pass::primitive::Tri;

#[derive(Debug)]
pub struct Clipper {
    //
}

impl Clipper {
    pub fn clip(&self, tri: Tri) -> impl Iterator<Item = Tri> {
        let clips = tri.0.map(|v| {
            !(v.x >= -v.w && v.x <= v.w && v.y >= -v.w && v.y <= v.w && v.z >= -v.w && v.z <= v.w)
        });

        let any = clips.iter().any(|x| *x);
        //let all = clips.iter().all(|x| *x);

        if any {
            tracing::debug!(?tri, "clipped");
        }

        (!any).then_some(tri).into_iter()
    }
}
