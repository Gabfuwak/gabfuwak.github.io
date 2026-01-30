export interface Mesh {
  positions: Float32Array<ArrayBuffer>;
  normals: Float32Array<ArrayBuffer>;
  uvs: Float32Array<ArrayBuffer>;
  colors: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
}

export function createSphere(
  radius: number,
  latitudeRes: number,
  longitudeRes: number,
  color: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0]
): Mesh {
        const positions: number[] = [];
        const normals: number[] = [];
        const uvs: number[] = [];
        const indices: number[] = [];

        const colors: number[] = [];

        for (let lat = 0; lat <= latitudeRes; lat++) {
          const theta = lat * Math.PI / latitudeRes;
          const sinTheta = Math.sin(theta);
          const cosTheta = Math.cos(theta);

          for (let lon = 0; lon <= longitudeRes; lon++) {
            const phi = lon * 2 * Math.PI / longitudeRes;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const x = cosPhi * sinTheta;
            const y = cosTheta;
            const z = sinPhi * sinTheta;

            positions.push(radius * x, radius * y, radius * z);
            normals.push(x, y, z);
            uvs.push(lon / longitudeRes, 1 - lat / latitudeRes);
            colors.push(...color);
          }
        }
      
        for (let lat = 0; lat < latitudeRes; lat++) {
          for (let lon = 0; lon < longitudeRes; lon++) {
            const first = lat * (longitudeRes + 1) + lon;
            const second = first + longitudeRes + 1;

            indices.push(
              first, second, first + 1,
              second, second + 1, first + 1
            );
          }
        }

        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;
      }


export function createBox(
  width: number,
  height: number,
  length: number,
  color: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0]
): Mesh {
        const w = width/2.0;
        const h = height/2.0;
        const l = length/2.0;
        const positions = [
          -w, -h,  l,
           w, -h,  l,
           w,  h,  l,
          -w,  h,  l,
          -w, -h, -l,
           w, -h, -l,
           w,  h, -l,
          -w,  h, -l,
        ];
        const den = Math.sqrt (w*w + h*h + l*l);
        const wn = w/den;
        const hn = h/den;
        const ln = l/den;
        const normals = [
          -wn, -hn,  ln,
           wn, -hn,  ln,
           wn,  hn,  ln,
          -wn,  hn,  ln,
          -wn, -hn, -ln,
           wn, -hn, -ln,
           wn,  hn, -ln,
          -wn,  hn, -ln,
        ];
        const uvs = [
          0.375, 0.750, 
          0.625, 0.750, 
          0.625, 1.0, 
          0.375, 1.0, 
          0.375, 0.250, 
          0.625, 0.250,  
          0.625, 0.5, 
          0.375, 0.5, 
        ];
        const indices = [
          0, 1, 2,
          0, 2, 3,
          1, 5, 6,
          1, 6, 2,
          5, 4, 7,
          5, 7, 6,
          4, 0, 3,
          4, 3, 7,
          3, 2, 6,
          3, 6, 7,
          4, 5, 1,
          4, 1, 0,
        ];
        const colors = new Array(8).fill(color).flat();
        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;
      }

function create_quad(
  a: number[],
  b: number[],
  c: number[],
  d: number[],
  color: [number, number, number, number]
): Mesh {
          const cross = (a: Float32Array | number[], b: Float32Array | number[]): Float32Array => {
            const dst = new Float32Array(3);

            const t0 = a[1] * b[2] - a[2] * b[1];
            const t1 = a[2] * b[0] - a[0] * b[2];
            const t2 = a[0] * b[1] - a[1] * b[0];

            dst[0] = t0;
            dst[1] = t1;
            dst[2] = t2;

            return dst;
          }

          const subtract = (a: number[], b: number[]): Float32Array => {
            const dst = new Float32Array(3);

            dst[0] = a[0] - b[0];
            dst[1] = a[1] - b[1];
            dst[2] = a[2] - b[2];

            return dst;
          }



          const positions = [
              a,
              b,
              c,
              d
          ].flat();

          // to verify
          const normal = cross(subtract(a, b), subtract(a, d));

        const normals = new Array(4).fill(normal).flat();

          const indices: number[] = [
              0, 1, 2,
              1, 2, 3,
          ]

          const uvs: number[] = [0,0, 1,0, 1,1, 0,1];

          const colors = [
              color,
              color,
              color,
              color
          ].flat()

        const mesh: Mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
        return mesh;

      }

export function createCornellBox(): Mesh {

        const floor = create_quad(
                          // a, b, c, d
                          [552.8, 0.0,   0.0],
                          [0.0, 0.0,   0.0],
                          [0.0, 0.0, 559.2],
                          [549.6, 0.0, 559.2],
                          // color
                          [1.0, 1.0, 1.0, 1.0] 
                        );

        /*const ceiling = create_quad(

                          [556.0, 548.8, 0.0   ],
                          [556.0, 548.8, 559.2],
                          [0.0, 548.8, 559.2],
                          [0.0, 548.8,   0.0],

                        );


        const back = create_quad(
[549.6,   0.0, 559.2], 
[  0.0,   0.0, 559.2],
[  0.0, 548.8, 559.2],
[556.0, 548.8, 559.2],
        )

        const right = 
          */
        



        /*const mesh = {
          positions: new Float32Array(positions),
          normals: new Float32Array(normals),
          uvs: new Float32Array(uvs),
          colors: new Float32Array(colors),
          indices: new Uint32Array(indices)
        };
      */
        return floor;
      }
